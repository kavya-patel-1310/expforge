"""Config-driven training engine. Every run is tracked end-to-end through the SDK:
params -> learning curves -> validation/test metrics -> evaluation report -> model artifact.
"""
from __future__ import annotations

import copy
import logging
import time
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from expforge.evaluation import metrics as E
from expforge.sdk import Client, Run
from expforge.training.config import validate_config
from expforge.training.datasets import Dataset, load_dataset
from expforge.training.models import MODELS, build_model

log = logging.getLogger("expforge.trainer")


def split_dataset(ds: Dataset, test_size: float, val_size: float, seed: int):
    strat = ds.y if ds.task == "classification" else None
    X_rest, X_test, y_rest, y_test = train_test_split(ds.X, ds.y, test_size=test_size, random_state=seed, stratify=strat)
    rel_val = val_size / (1 - test_size)
    strat = y_rest if ds.task == "classification" else None
    X_tr, X_val, y_tr, y_val = train_test_split(X_rest, y_rest, test_size=rel_val, random_state=seed, stratify=strat)
    return X_tr, X_val, X_test, y_tr, y_val, y_test


def _scores(task: str, model, X, y, n_classes: int, prefix: str) -> dict[str, float]:
    pred = model.predict(X)
    if task == "classification":
        proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        if proba is not None and proba.shape[1] != n_classes:  # pad if a class was unseen in training
            full = np.zeros((len(X), n_classes))
            full[:, model.classes_.astype(int)] = proba
            proba = full
        return E.classification_metrics(y, pred, proba, prefix)
    return E.regression_metrics(y, pred, prefix)


def _loss(task: str, model, X, y, n_classes: int) -> float:
    if task == "classification":
        from sklearn.metrics import log_loss
        return float(log_loss(y, model.predict_proba(X), labels=np.arange(n_classes)))
    from sklearn.metrics import mean_squared_error
    return float(mean_squared_error(y, model.predict(X)))


# --------------------------------------------------------------------------- training strategies

def _train_partial_fit(est, task, X_tr, y_tr, X_val, y_val, n_classes, cfg, run: Run) -> tuple[Any, dict]:
    """Epoch loop with early stopping on validation loss; restores the best checkpoint."""
    epochs, patience = int(cfg["epochs"]), int(cfg["early_stopping_patience"])
    primary = "accuracy" if task == "classification" else "r2"
    rng = np.random.default_rng(0)
    best_loss, best_est, best_epoch, bad = np.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X_tr))
        if task == "classification":
            est.partial_fit(X_tr[order], y_tr[order], classes=np.arange(n_classes))
        else:
            est.partial_fit(X_tr[order], y_tr[order])
        tr_loss, val_loss = _loss(task, est, X_tr, y_tr, n_classes), _loss(task, est, X_val, y_val, n_classes)
        tr_s = _scores(task, est, X_tr, y_tr, n_classes, "")[primary]
        val_s = _scores(task, est, X_val, y_val, n_classes, "")[primary]
        if epoch % int(cfg["log_every"]) == 0:
            run.log({"train_loss": tr_loss, "val_loss": val_loss, f"train_{primary}": tr_s, f"val_{primary}": val_s},
                    step=epoch)
        if val_loss < best_loss - 1e-5:
            best_loss, best_est, best_epoch, bad = val_loss, copy.deepcopy(est), epoch, 0
        else:
            bad += 1
            if patience and bad >= patience:
                log.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                break
    return best_est or est, {"epochs_trained": epoch, "best_epoch": best_epoch, "best_val_loss": round(best_loss, 6),
                             "early_stopped": bool(patience and epoch < epochs)}


def _train_staged(est, task, X_tr, y_tr, X_val, y_val, n_classes, cfg, run: Run) -> tuple[Any, dict]:
    """Boosting: fit once, then replay staged predictions to recover per-iteration learning curves."""
    from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
    est.fit(X_tr, y_tr)
    n = est.n_estimators_
    every = max(1, n // 60)
    best_loss, best_iter = np.inf, 0
    if task == "classification":
        stages = zip(est.staged_predict_proba(X_tr), est.staged_predict_proba(X_val), strict=True)
        for i, (p_tr, p_val) in enumerate(stages, 1):
            vl = log_loss(y_val, p_val, labels=np.arange(n_classes))
            if vl < best_loss:
                best_loss, best_iter = vl, i
            if i % every == 0 or i == n:
                run.log({"train_loss": log_loss(y_tr, p_tr, labels=np.arange(n_classes)), "val_loss": vl,
                         "train_accuracy": accuracy_score(y_tr, p_tr.argmax(1)),
                         "val_accuracy": accuracy_score(y_val, p_val.argmax(1))}, step=i)
    else:
        for i, (p_tr, p_val) in enumerate(zip(est.staged_predict(X_tr), est.staged_predict(X_val), strict=True), 1):
            vl = mean_squared_error(y_val, p_val)
            if vl < best_loss:
                best_loss, best_iter = vl, i
            if i % every == 0 or i == n:
                run.log({"train_loss": mean_squared_error(y_tr, p_tr), "val_loss": vl,
                         "train_r2": r2_score(y_tr, p_tr), "val_r2": r2_score(y_val, p_val)}, step=i)
    return est, {"epochs_trained": int(n), "best_epoch": int(best_iter), "best_val_loss": round(float(best_loss), 6)}


def _train_single(est, task, X_tr, y_tr, X_val, y_val, n_classes, cfg, run: Run) -> tuple[Any, dict]:
    info: dict[str, Any] = {}
    folds = int(cfg.get("cv_folds") or 0)
    if folds > 1:
        splitter = (StratifiedKFold if task == "classification" else KFold)(folds, shuffle=True, random_state=0)
        scoring = "f1_macro" if task == "classification" else "neg_root_mean_squared_error"
        scores = cross_val_score(copy.deepcopy(est), X_tr, y_tr, cv=splitter, scoring=scoring, n_jobs=1)
        scores = -scores if scoring.startswith("neg_") else scores
        for i, s in enumerate(scores, 1):
            run.log({f"cv_{scoring.removeprefix('neg_')}": float(s)}, step=i)
        info.update({"cv_metric": scoring.removeprefix("neg_"), "cv_mean": round(float(scores.mean()), 6),
                     "cv_std": round(float(scores.std()), 6), "cv_folds": folds})
    est.fit(X_tr, y_tr)
    return est, info


STRATEGIES = {"partial_fit": _train_partial_fit, "staged": _train_staged, None: _train_single}


# --------------------------------------------------------------------------- public API

def run_single(config: dict[str, Any], client: Client, sweep_id: int | None = None,
               run_name: str | None = None, dataset: Dataset | None = None) -> int:
    cfg = validate_config(config)
    task, ds_cfg, model_cfg, train_cfg = cfg["task"], cfg["dataset"], cfg["model"], cfg["training"]
    ds = dataset or load_dataset(ds_cfg["name"], **ds_cfg.get("params", {}))
    X_tr, X_val, X_test, y_tr, y_val, y_test = split_dataset(ds, ds_cfg["test_size"], ds_cfg["val_size"], ds_cfg["seed"])
    n_classes = len(np.unique(ds.y)) if task == "classification" else 0

    est, model_params = build_model(model_cfg["name"], task, model_cfg.get("params"))
    flat_params = {"model": model_cfg["name"], "dataset": ds_cfg["name"], "scale": train_cfg["scale"],
                   "seed": ds_cfg["seed"], "test_size": ds_cfg["test_size"], "val_size": ds_cfg["val_size"],
                   **{k: v for k, v in model_params.items() if k not in ("n_jobs", "verbose")}}
    tags = {"task": task, "model_family": model_cfg["name"], **cfg.get("tags", {})}
    if sweep_id is not None:
        tags["sweep_id"] = sweep_id

    run = client.start_run(cfg["experiment"], name=run_name or cfg.get("name"), params=flat_params, tags=tags,
                           sweep_id=sweep_id, extra_system_info=ds.describe())
    with run:
        scaler = StandardScaler().fit(X_tr) if train_cfg["scale"] else None
        tf = (lambda a: scaler.transform(a)) if scaler else (lambda a: a)
        strategy = STRATEGIES[MODELS[model_cfg["name"]].iterative]

        t0 = time.perf_counter()
        fitted, train_info = strategy(est, task, tf(X_tr), y_tr, tf(X_val), y_val, n_classes, train_cfg, run)
        train_time = time.perf_counter() - t0
        model = Pipeline([("scaler", scaler), ("model", fitted)]) if scaler else fitted

        summary: dict[str, Any] = {"train_time_s": round(train_time, 4), "n_train": len(X_tr), "n_val": len(X_val),
                                   "n_test": len(X_test), **train_info}
        summary.update(_scores(task, model, X_tr, y_tr, n_classes, "train_"))
        summary.update(_scores(task, model, X_val, y_val, n_classes, "val_"))
        summary.update(_scores(task, model, X_test, y_test, n_classes, "test_"))
        summary.update(E.inference_profile(model, X_test))
        if task == "classification":
            summary["overfit_gap"] = round(summary["train_accuracy"] - summary["val_accuracy"], 6)
        else:
            summary["overfit_gap"] = round(summary["train_r2"] - summary["val_r2"], 6)
        run.log_summary(summary)

        test_pred = model.predict(X_test)
        if task == "classification":
            proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            report = E.classification_report_payload(y_test, test_pred, proba, ds.target_names)
        else:
            report = E.regression_report_payload(y_test, test_pred)
        report["feature_importance"] = E.feature_importance(
            model, X_val, y_val, ds.feature_names, use_permutation=train_cfg["permutation_importance"])
        report["dataset"] = ds.describe()
        report["config"] = cfg
        run.log_evaluation(report)
        run.log_model(model)
    return run.id


def run_from_config(config: dict[str, Any], client: Client) -> dict[str, Any]:
    """Entry point used by the CLI and the server's job worker."""
    cfg = validate_config(config)
    if cfg.get("sweep"):
        from expforge.training.sweep import run_sweep
        result = run_sweep(cfg, client)
    else:
        rid = run_single(cfg, client)
        result = {"run_ids": [rid], "best_run_id": rid, "sweep_id": None}

    reg = cfg.get("register")
    if reg and result.get("best_run_id"):
        model = client.register_model(reg["name"], result["best_run_id"])
        version = model["versions"][-1]["version"]
        if reg.get("stage"):
            client.set_stage(reg["name"], version, reg["stage"])
        result["registered"] = {"name": reg["name"], "version": version, "stage": reg.get("stage", "None")}
    return result
