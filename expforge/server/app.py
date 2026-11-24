"""ExpForge tracking server: REST API + static dashboard.

Run with:  expforge server --port 8000      (or)  uvicorn expforge.server.app:create_app --factory
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from expforge import __version__
from expforge.server import schemas as S
from expforge.server.db import (
    Artifact,
    Experiment,
    Job,
    Metric,
    ModelVersion,
    RegisteredModel,
    Run,
    Sweep,
    default_home,
    make_engine,
    make_session_factory,
    utcnow,
)

log = logging.getLogger("expforge.server")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# --------------------------------------------------------------------------- helpers

def _duration(run: Run) -> float | None:
    end = run.ended_at or (utcnow() if run.status == "running" else None)
    return round((end - run.started_at).total_seconds(), 2) if end else None


def _run_out(run: Run) -> S.RunOut:
    out = S.RunOut.model_validate(run)
    out.experiment_name = run.experiment.name if run.experiment else ""
    out.duration_s = _duration(run)
    return out


def _get_or_404(db: Session, model, obj_id: int):
    obj = db.get(model, obj_id)
    if obj is None:
        raise HTTPException(404, f"{model.__name__} {obj_id} not found")
    return obj


class ModelCache:
    """Small thread-safe LRU of deserialized models so /predict doesn't hit disk every call."""

    def __init__(self, capacity: int = 8):
        self.capacity, self._items, self._lock = capacity, OrderedDict(), threading.Lock()

    def get(self, path: str):
        import joblib
        with self._lock:
            if path in self._items:
                self._items.move_to_end(path)
                return self._items[path]
        model = joblib.load(path)  # artifacts are written by ExpForge itself; only load trusted stores
        with self._lock:
            self._items[path] = model
            if len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return model


# --------------------------------------------------------------------------- app factory

def create_app(db_url: str | None = None, artifact_root: str | Path | None = None,
               transport_factory: Callable[[], Any] | None = None, job_workers: int = 2) -> FastAPI:
    engine = make_engine(db_url)
    SessionLocal = make_session_factory(engine)
    artifact_root = Path(artifact_root or os.environ.get("EXPFORGE_ARTIFACTS", default_home() / "artifacts"))
    artifact_root.mkdir(parents=True, exist_ok=True)
    executor = ThreadPoolExecutor(max_workers=job_workers, thread_name_prefix="expforge-job")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="ExpForge", version=__version__, lifespan=lifespan,
                  description="Experiment tracking, model evaluation, and model registry for ML teams.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.session_factory = SessionLocal
    app.state.artifact_root = artifact_root
    app.state.model_cache = ModelCache()
    app.state.executor = executor
    if transport_factory is None:  # background jobs report back to this same server over HTTP
        from expforge.sdk import HttpTransport
        self_url = os.environ.get("EXPFORGE_SELF_URL", "http://127.0.0.1:8000")
        transport_factory = lambda: HttpTransport(self_url)  # noqa: E731
    app.state.transport_factory = transport_factory

    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    # ------------------------------------------------------------------ health / stats
    @app.get("/api/health", tags=["meta"])
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/stats", tags=["meta"])
    def stats(db: Session = Depends(get_db)):
        by_status = dict(db.execute(select(Run.status, func.count()).group_by(Run.status)).all())
        return {
            "experiments": db.scalar(select(func.count(Experiment.id))),
            "runs": sum(by_status.values()),
            "runs_by_status": by_status,
            "metric_points": db.scalar(select(func.count(Metric.id))),
            "registered_models": db.scalar(select(func.count(RegisteredModel.id))),
            "sweeps": db.scalar(select(func.count(Sweep.id))),
        }

    # ------------------------------------------------------------------ experiments
    @app.post("/api/experiments", response_model=S.ExperimentOut, tags=["experiments"])
    def create_experiment(body: S.ExperimentCreate, db: Session = Depends(get_db)):
        if db.scalar(select(Experiment).where(Experiment.name == body.name)):
            raise HTTPException(409, "experiment already exists")
        exp = Experiment(**body.model_dump())
        db.add(exp)
        db.commit()
        return S.ExperimentOut.model_validate(exp)

    @app.get("/api/experiments", response_model=list[S.ExperimentOut], tags=["experiments"])
    def list_experiments(db: Session = Depends(get_db)):
        counts = dict(db.execute(select(Run.experiment_id, func.count()).group_by(Run.experiment_id)).all())
        out = []
        for exp in db.scalars(select(Experiment).order_by(Experiment.created_at.desc())):
            item = S.ExperimentOut.model_validate(exp)
            item.run_count = counts.get(exp.id, 0)
            out.append(item)
        return out

    @app.get("/api/experiments/{exp_id}", response_model=S.ExperimentOut, tags=["experiments"])
    def get_experiment(exp_id: int, db: Session = Depends(get_db)):
        exp = _get_or_404(db, Experiment, exp_id)
        item = S.ExperimentOut.model_validate(exp)
        item.run_count = len(exp.runs)
        return item

    @app.delete("/api/experiments/{exp_id}", tags=["experiments"])
    def delete_experiment(exp_id: int, db: Session = Depends(get_db)):
        exp = _get_or_404(db, Experiment, exp_id)
        run_ids = [r.id for r in exp.runs]
        db.delete(exp)
        db.commit()
        for rid in run_ids:
            shutil.rmtree(artifact_root / str(rid), ignore_errors=True)
        return {"deleted": exp_id}

    # ------------------------------------------------------------------ runs
    @app.post("/api/runs", response_model=S.RunOut, tags=["runs"])
    def create_run(body: S.RunCreate, db: Session = Depends(get_db)):
        exp = db.scalar(select(Experiment).where(Experiment.name == body.experiment))
        if exp is None:
            exp = Experiment(name=body.experiment)
            db.add(exp)
            db.flush()
        count = db.scalar(select(func.count(Run.id)).where(Run.experiment_id == exp.id)) or 0
        run = Run(experiment_id=exp.id, name=body.name or f"run-{count + 1}", params=body.params,
                  tags=body.tags, system_info=body.system_info, sweep_id=body.sweep_id, notes=body.notes)
        db.add(run)
        db.commit()
        db.refresh(run)
        return _run_out(run)

    @app.get("/api/runs", response_model=list[S.RunOut], tags=["runs"])
    def list_runs(experiment_id: int | None = None, sweep_id: int | None = None, status: str | None = None,
                  tag: str | None = Query(None, description="key=value"), limit: int = Query(500, le=5000),
                  db: Session = Depends(get_db)):
        q = select(Run).order_by(Run.started_at.desc()).limit(limit)
        if experiment_id is not None:
            q = q.where(Run.experiment_id == experiment_id)
        if sweep_id is not None:
            q = q.where(Run.sweep_id == sweep_id)
        if status:
            q = q.where(Run.status == status)
        runs = list(db.scalars(q))
        if tag and "=" in tag:
            k, v = tag.split("=", 1)
            runs = [r for r in runs if str(r.tags.get(k)) == v]
        return [_run_out(r) for r in runs]

    @app.get("/api/runs/{run_id}", response_model=S.RunDetail, tags=["runs"])
    def get_run(run_id: int, db: Session = Depends(get_db)):
        run = _get_or_404(db, Run, run_id)
        detail = S.RunDetail.model_validate(run)
        detail.experiment_name = run.experiment.name
        detail.duration_s = _duration(run)
        detail.metric_keys = sorted(db.scalars(select(Metric.key).where(Metric.run_id == run_id).distinct()))
        return detail

    @app.patch("/api/runs/{run_id}", response_model=S.RunOut, tags=["runs"])
    def update_run(run_id: int, body: S.RunUpdate, db: Session = Depends(get_db)):
        run = _get_or_404(db, Run, run_id)
        data = body.model_dump(exclude_none=True)
        for merge_key in ("params", "tags", "summary", "evaluation"):
            if merge_key in data:  # JSON columns are merged, not replaced
                data[merge_key] = {**(getattr(run, merge_key) or {}), **data[merge_key]}
        for k, v in data.items():
            setattr(run, k, v)
        if body.status in ("finished", "failed", "killed") and run.ended_at is None:
            run.ended_at = utcnow()
        db.commit()
        db.refresh(run)
        return _run_out(run)

    @app.delete("/api/runs/{run_id}", tags=["runs"])
    def delete_run(run_id: int, db: Session = Depends(get_db)):
        db.delete(_get_or_404(db, Run, run_id))
        db.commit()
        shutil.rmtree(artifact_root / str(run_id), ignore_errors=True)
        return {"deleted": run_id}

    # ------------------------------------------------------------------ metrics
    @app.post("/api/runs/{run_id}/metrics", tags=["metrics"])
    def log_metrics(run_id: int, body: S.MetricBatch, db: Session = Depends(get_db)):
        _get_or_404(db, Run, run_id)
        rows = [Metric(run_id=run_id, key=m.key, step=m.step, value=float(m.value))
                for m in body.metrics if np.isfinite(m.value)]
        db.add_all(rows)
        db.commit()
        return {"logged": len(rows)}

    @app.get("/api/runs/{run_id}/metrics", tags=["metrics"])
    def get_metrics(run_id: int, keys: str | None = Query(None, description="comma-separated"),
                    db: Session = Depends(get_db)) -> dict[str, list[dict[str, float]]]:
        _get_or_404(db, Run, run_id)
        q = select(Metric.key, Metric.step, Metric.value).where(Metric.run_id == run_id).order_by(Metric.step, Metric.id)
        if keys:
            q = q.where(Metric.key.in_(keys.split(",")))
        series: dict[str, list[dict[str, float]]] = {}
        for key, step, value in db.execute(q):
            series.setdefault(key, []).append({"step": step, "value": value})
        return series

    # ------------------------------------------------------------------ artifacts
    @app.post("/api/runs/{run_id}/artifacts", response_model=S.ArtifactOut, tags=["artifacts"])
    def upload_artifact(run_id: int, file: UploadFile = File(...), kind: str = Form("file"),
                        db: Session = Depends(get_db)):
        _get_or_404(db, Run, run_id)
        safe_name = Path(file.filename or "artifact.bin").name
        dest_dir = artifact_root / str(run_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / safe_name
        h, size = hashlib.sha256(), 0
        with dest.open("wb") as fh:
            while chunk := file.file.read(1 << 20):
                h.update(chunk)
                size += len(chunk)
                fh.write(chunk)
        existing = db.scalar(select(Artifact).where(Artifact.run_id == run_id, Artifact.name == safe_name))
        art = existing or Artifact(run_id=run_id, name=safe_name)
        art.kind, art.path, art.size_bytes, art.sha256 = kind, str(dest), size, h.hexdigest()
        db.add(art)
        db.commit()
        return S.ArtifactOut.model_validate(art)

    @app.get("/api/artifacts/{artifact_id}/download", tags=["artifacts"])
    def download_artifact(artifact_id: int, db: Session = Depends(get_db)):
        art = _get_or_404(db, Artifact, artifact_id)
        if not Path(art.path).exists():
            raise HTTPException(410, "artifact file missing from store")
        return FileResponse(art.path, filename=art.name)

    # ------------------------------------------------------------------ comparison & leaderboard
    @app.post("/api/compare", tags=["analysis"])
    def compare(body: S.CompareRequest, db: Session = Depends(get_db)):
        runs = [_get_or_404(db, Run, rid) for rid in body.run_ids]
        all_params = sorted({k for r in runs for k in r.params})
        all_metrics = sorted({k for r in runs for k in r.summary if isinstance(r.summary.get(k), (int, float))})
        params_table = {k: [r.params.get(k) for r in runs] for k in all_params}
        differing = [k for k, vals in params_table.items() if len({repr(v) for v in vals}) > 1]
        metrics_table = {k: [r.summary.get(k) for r in runs] for k in all_metrics}
        best = {}
        for k, vals in metrics_table.items():
            nums = [(v, r.id) for v, r in zip(vals, runs, strict=True) if isinstance(v, (int, float))]
            if nums:
                lower_better = any(t in k.lower() for t in ("loss", "error", "mae", "mse", "rmse", "time"))
                best[k] = (min if lower_better else max)(nums)[1]
        return {
            "runs": [_run_out(r).model_dump(mode="json") for r in runs],
            "params": params_table, "differing_params": differing,
            "metrics": metrics_table, "best_run_per_metric": best,
            "series": {r.id: get_metrics(r.id, None, db) for r in runs},
        }

    @app.get("/api/experiments/{exp_id}/leaderboard", tags=["analysis"])
    def leaderboard(exp_id: int, metric: str, direction: str = Query("maximize", pattern="^(maximize|minimize)$"),
                    limit: int = 20, db: Session = Depends(get_db)):
        _get_or_404(db, Experiment, exp_id)
        runs = [r for r in db.scalars(select(Run).where(Run.experiment_id == exp_id, Run.status == "finished"))
                if isinstance(r.summary.get(metric), (int, float))]
        runs.sort(key=lambda r: r.summary[metric], reverse=direction == "maximize")
        return [{"rank": i + 1, "run_id": r.id, "name": r.name, "value": r.summary[metric],
                 "model": r.params.get("model"), "params": r.params} for i, r in enumerate(runs[:limit])]

    # ------------------------------------------------------------------ sweeps
    @app.post("/api/sweeps", response_model=S.SweepOut, tags=["sweeps"])
    def create_sweep(body: dict[str, Any], db: Session = Depends(get_db)):
        exp = db.scalar(select(Experiment).where(Experiment.name == body["experiment"]))
        if exp is None:
            exp = Experiment(name=body["experiment"])
            db.add(exp)
            db.flush()
        sweep = Sweep(experiment_id=exp.id, name=body.get("name", "sweep"), strategy=body.get("strategy", "grid"),
                      objective=body["objective"], direction=body.get("direction", "maximize"),
                      search_space=body.get("search_space", {}))
        db.add(sweep)
        db.commit()
        return S.SweepOut.model_validate(sweep)

    @app.get("/api/sweeps", response_model=list[S.SweepOut], tags=["sweeps"])
    def list_sweeps(experiment_id: int | None = None, db: Session = Depends(get_db)):
        q = select(Sweep).order_by(Sweep.created_at.desc())
        if experiment_id is not None:
            q = q.where(Sweep.experiment_id == experiment_id)
        return [S.SweepOut.model_validate(s) for s in db.scalars(q)]

    @app.patch("/api/sweeps/{sweep_id}", response_model=S.SweepOut, tags=["sweeps"])
    def update_sweep(sweep_id: int, body: dict[str, Any], db: Session = Depends(get_db)):
        sweep = _get_or_404(db, Sweep, sweep_id)
        for k in ("status", "best_run_id"):
            if k in body:
                setattr(sweep, k, body[k])
        db.commit()
        return S.SweepOut.model_validate(sweep)

    # ------------------------------------------------------------------ model registry
    @app.post("/api/models", response_model=S.RegisteredModelOut, tags=["registry"])
    def register_model(body: S.RegisterModelRequest, db: Session = Depends(get_db)):
        run = _get_or_404(db, Run, body.run_id)
        art = db.scalar(select(Artifact).where(Artifact.run_id == run.id, Artifact.name == body.artifact_name))
        if art is None:
            raise HTTPException(400, f"run {run.id} has no artifact named {body.artifact_name!r}")
        model = db.scalar(select(RegisteredModel).where(RegisteredModel.name == body.name))
        if model is None:
            model = RegisteredModel(name=body.name, description=body.description)
            db.add(model)
            db.flush()
        next_version = (max((v.version for v in model.versions), default=0)) + 1
        db.add(ModelVersion(model_id=model.id, version=next_version, run_id=run.id, artifact_id=art.id,
                            metrics_snapshot={k: v for k, v in run.summary.items() if isinstance(v, (int, float))}))
        db.commit()
        db.refresh(model)
        return S.RegisteredModelOut.model_validate(model)

    @app.get("/api/models", response_model=list[S.RegisteredModelOut], tags=["registry"])
    def list_models(db: Session = Depends(get_db)):
        return [S.RegisteredModelOut.model_validate(m) for m in db.scalars(select(RegisteredModel).order_by(RegisteredModel.name))]

    @app.patch("/api/models/{name}/versions/{version}", response_model=S.ModelVersionOut, tags=["registry"])
    def set_stage(name: str, version: int, body: S.StageUpdate, db: Session = Depends(get_db)):
        model = db.scalar(select(RegisteredModel).where(RegisteredModel.name == name))
        if model is None:
            raise HTTPException(404, "model not found")
        target = next((v for v in model.versions if v.version == version), None)
        if target is None:
            raise HTTPException(404, "version not found")
        if body.archive_existing and body.stage in ("Production", "Staging"):
            for v in model.versions:  # only one version may hold a given stage
                if v.stage == body.stage and v.id != target.id:
                    v.stage = "Archived"
        target.stage = body.stage
        db.commit()
        return S.ModelVersionOut.model_validate(target)

    @app.post("/api/models/{name}/predict", tags=["serving"])
    def predict(name: str, body: S.PredictRequest, request: Request, db: Session = Depends(get_db)):
        model = db.scalar(select(RegisteredModel).where(RegisteredModel.name == name))
        if model is None:
            raise HTTPException(404, "model not found")
        if body.version is not None:
            mv = next((v for v in model.versions if v.version == body.version), None)
        else:
            mv = next((v for v in reversed(model.versions) if v.stage == "Production"), None)
        if mv is None:
            raise HTTPException(409, "no matching version (promote one to Production or pass `version`)")
        art = db.get(Artifact, mv.artifact_id)
        estimator = request.app.state.model_cache.get(art.path)
        X = np.asarray(body.instances, dtype=float)
        try:
            preds = estimator.predict(X)
        except ValueError as exc:
            raise HTTPException(422, f"bad input shape: {exc}") from exc
        out: dict[str, Any] = {"model": name, "version": mv.version, "predictions": np.asarray(preds).tolist()}
        if body.return_proba and hasattr(estimator, "predict_proba"):
            out["probabilities"] = np.round(estimator.predict_proba(X), 6).tolist()
        return out

    # ------------------------------------------------------------------ training jobs
    def _execute_job(job_id: int):
        from expforge.sdk import Client
        from expforge.training.trainer import run_from_config
        db = SessionLocal()
        job = db.get(Job, job_id)
        job.status = "running"
        db.commit()
        try:
            result = run_from_config(job.config, client=Client(transport=app.state.transport_factory()))
            job.run_ids, job.sweep_id, job.status = result["run_ids"], result.get("sweep_id"), "finished"
        except Exception as exc:  # noqa: BLE001 - surface any training failure on the job
            log.exception("job %s failed", job_id)
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
        job.finished_at = utcnow()
        db.commit()
        db.close()

    @app.post("/api/jobs", response_model=S.JobOut, tags=["jobs"])
    def create_job(body: S.JobCreate, db: Session = Depends(get_db)):
        from expforge.training.config import validate_config
        try:
            config = validate_config(body.config)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        job = Job(config=config)
        db.add(job)
        db.commit()
        app.state.executor.submit(_execute_job, job.id)
        return S.JobOut.model_validate(job)

    @app.get("/api/jobs", response_model=list[S.JobOut], tags=["jobs"])
    def list_jobs(db: Session = Depends(get_db)):
        return [S.JobOut.model_validate(j) for j in db.scalars(select(Job).order_by(Job.created_at.desc()).limit(50))]

    @app.get("/api/jobs/{job_id}", response_model=S.JobOut, tags=["jobs"])
    def get_job(job_id: int, db: Session = Depends(get_db)):
        return S.JobOut.model_validate(_get_or_404(db, Job, job_id))

    @app.get("/api/catalog", tags=["jobs"])
    def catalog():
        from expforge.training.datasets import DATASETS
        from expforge.training.models import MODELS
        return {
            "datasets": {k: {"task": v.task, "description": v.description} for k, v in DATASETS.items()},
            "models": {k: {"tasks": sorted(v.tasks), "defaults": v.defaults, "iterative": v.iterative}
                       for k, v in MODELS.items()},
        }

    # ------------------------------------------------------------------ dashboard
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui/")

    return app
