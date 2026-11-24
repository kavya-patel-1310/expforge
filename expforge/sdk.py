"""Python client SDK.

    import expforge

    with expforge.init(experiment="churn", params={"lr": 0.01}) as run:
        for epoch in range(10):
            run.log({"train_loss": loss, "val_acc": acc}, step=epoch)
        run.log_model(model)

Metrics are buffered and flushed in batches so tight training loops don't block on HTTP.
"""
from __future__ import annotations

import io
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import requests

DEFAULT_URL = "http://127.0.0.1:8000"


class ExpForgeError(RuntimeError):
    pass


class HttpTransport:
    """Thin HTTP layer. `session` can be a requests.Session or anything with the same `.request`
    signature (e.g. FastAPI's TestClient), which is how the test-suite runs the SDK in-process."""

    def __init__(self, base_url: str = DEFAULT_URL, session: Any = None, timeout: float = 30.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout, self.retries = timeout, retries

    def request(self, method: str, path: str, **kwargs) -> Any:
        if isinstance(self.session, requests.Session):
            kwargs.setdefault("timeout", self.timeout)
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.session.request(method, self.base_url + path, **kwargs)
            except requests.ConnectionError as exc:  # transient: back off and retry
                last_exc = exc
                time.sleep(0.25 * 2 ** attempt)
                continue
            if resp.status_code >= 500 and attempt < self.retries - 1:
                time.sleep(0.25 * 2 ** attempt)
                continue
            if resp.status_code >= 400:
                raise ExpForgeError(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}")
            return resp.json() if resp.content else None
        raise ExpForgeError(f"could not reach ExpForge server at {self.base_url}: {last_exc}")


def collect_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0], "platform": platform.platform(), "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(), "argv": " ".join(sys.argv)[:300],
    }
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2)
        if sha.returncode == 0:
            info["git_commit"] = sha.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=2)
            info["git_dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    for lib in ("sklearn", "numpy", "pandas", "torch"):
        mod = sys.modules.get(lib)
        if mod is not None and hasattr(mod, "__version__"):
            info[f"{lib}_version"] = mod.__version__
    return info


class Run:
    """Handle to a single tracked run. Use as a context manager to auto-finish / mark failed."""

    def __init__(self, client: Client, data: dict[str, Any], flush_every: int = 50):
        self.client, self.data = client, data
        self.id: int = data["id"]
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._flush_every = flush_every
        self._finished = False

    def __repr__(self) -> str:
        return f"<Run id={self.id} name={self.data.get('name')!r}>"

    # metrics ---------------------------------------------------------------
    def log(self, metrics: Mapping[str, float], step: int = 0) -> None:
        with self._lock:
            # NaN/inf can't travel as JSON; drop them client-side (the server filters too)
            self._buffer.extend({"key": k, "value": float(v), "step": int(step)} for k, v in metrics.items()
                                if math.isfinite(float(v)))
            should_flush = len(self._buffer) >= self._flush_every
        if should_flush:
            self.flush()

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        self.log({key: value}, step)

    def flush(self) -> None:
        with self._lock:
            batch, self._buffer = self._buffer, []
        if batch:
            self.client._t.request("POST", f"/api/runs/{self.id}/metrics", json={"metrics": batch})

    # metadata --------------------------------------------------------------
    def _patch(self, **fields) -> None:
        self.data = self.client._t.request("PATCH", f"/api/runs/{self.id}", json=fields)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._patch(params=_jsonable(params))

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        self._patch(tags=_jsonable(tags))

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        """Final / headline numbers shown in tables, leaderboards and comparisons."""
        self._patch(summary=_jsonable(summary))

    def log_evaluation(self, report: Mapping[str, Any]) -> None:
        """Rich evaluation payload (confusion matrix, ROC/PR curves, feature importance...)."""
        self._patch(evaluation=_jsonable(report))

    # artifacts -------------------------------------------------------------
    def log_artifact(self, path: str | Path, kind: str = "file", name: str | None = None) -> dict[str, Any]:
        path = Path(path)
        with path.open("rb") as fh:
            return self.client._t.request("POST", f"/api/runs/{self.id}/artifacts",
                                          files={"file": (name or path.name, fh)}, data={"kind": kind})

    def log_bytes(self, payload: bytes, name: str, kind: str = "file") -> dict[str, Any]:
        return self.client._t.request("POST", f"/api/runs/{self.id}/artifacts",
                                      files={"file": (name, io.BytesIO(payload))}, data={"kind": kind})

    def log_model(self, estimator: Any, name: str = "model.joblib") -> dict[str, Any]:
        import joblib
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / name
            joblib.dump(estimator, p, compress=3)
            return self.log_artifact(p, kind="model")

    # lifecycle -------------------------------------------------------------
    def finish(self, status: str = "finished", error: str | None = None) -> None:
        if self._finished:
            return
        self.flush()
        fields: dict[str, Any] = {"status": status}
        if error:
            fields["error"] = error[:5000]
        self._patch(**fields)
        self._finished = True

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is KeyboardInterrupt:
            self.finish("killed")
        elif exc_type is not None:
            self.finish("failed", error=f"{exc_type.__name__}: {exc}")
        else:
            self.finish("finished")
        return False


class Client:
    def __init__(self, url: str | None = None, transport: Any = None):
        self._t = transport or HttpTransport(url or os.environ.get("EXPFORGE_URL", DEFAULT_URL))

    def start_run(self, experiment: str, name: str | None = None, params: Mapping[str, Any] | None = None,
                  tags: Mapping[str, Any] | None = None, sweep_id: int | None = None,
                  extra_system_info: Mapping[str, Any] | None = None) -> Run:
        body = {"experiment": experiment, "name": name, "params": _jsonable(params or {}),
                "tags": _jsonable(tags or {}), "sweep_id": sweep_id,
                "system_info": {**collect_system_info(), **_jsonable(extra_system_info or {})}}
        return Run(self, self._t.request("POST", "/api/runs", json=body))

    def create_sweep(self, experiment: str, name: str, strategy: str, objective: str, direction: str,
                     search_space: Mapping[str, Any]) -> dict[str, Any]:
        return self._t.request("POST", "/api/sweeps", json={
            "experiment": experiment, "name": name, "strategy": strategy, "objective": objective,
            "direction": direction, "search_space": _jsonable(search_space)})

    def update_sweep(self, sweep_id: int, **fields) -> dict[str, Any]:
        return self._t.request("PATCH", f"/api/sweeps/{sweep_id}", json=fields)

    def get_run(self, run_id: int) -> dict[str, Any]:
        return self._t.request("GET", f"/api/runs/{run_id}")

    def runs(self, experiment_id: int | None = None) -> list[dict[str, Any]]:
        q = f"?experiment_id={experiment_id}" if experiment_id is not None else ""
        return self._t.request("GET", f"/api/runs{q}")

    def compare(self, run_ids: list[int]) -> dict[str, Any]:
        return self._t.request("POST", "/api/compare", json={"run_ids": run_ids})

    def register_model(self, name: str, run_id: int, artifact_name: str = "model.joblib") -> dict[str, Any]:
        return self._t.request("POST", "/api/models", json={"name": name, "run_id": run_id, "artifact_name": artifact_name})

    def set_stage(self, name: str, version: int, stage: str) -> dict[str, Any]:
        return self._t.request("PATCH", f"/api/models/{name}/versions/{version}", json={"stage": stage})

    def predict(self, name: str, instances: list[list[float]], version: int | None = None,
                return_proba: bool = False) -> dict[str, Any]:
        return self._t.request("POST", f"/api/models/{name}/predict",
                               json={"instances": instances, "version": version, "return_proba": return_proba})


def init(experiment: str, name: str | None = None, params: Mapping[str, Any] | None = None,
         tags: Mapping[str, Any] | None = None, url: str | None = None) -> Run:
    """One-liner entry point mirroring the W&B / MLflow ergonomics."""
    return Client(url).start_run(experiment, name=name, params=params, tags=tags)


def _jsonable(obj: Any) -> Any:
    """Coerce numpy scalars / tuples / paths into JSON-safe values."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if getattr(obj, "ndim", 0) and hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
