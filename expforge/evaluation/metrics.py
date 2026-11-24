"""Model evaluation: scalar metrics plus chart-ready payloads (curves, matrices, importances).

Everything returned is plain JSON so it can be stored on the run and rendered by the dashboard.
"""
from __future__ import annotations

import pickle
import time
from typing import Any

import numpy as np
from sklearn import metrics as M
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize

MAX_CURVE_POINTS = 120


def _downsample(*arrays: np.ndarray, n: int = MAX_CURVE_POINTS) -> list[list[float]]:
    length = len(arrays[0])
    idx = np.unique(np.linspace(0, length - 1, min(n, length)).astype(int))
    return [np.round(a[idx], 5).tolist() for a in arrays]


def _r(x: float) -> float:
    return float(round(float(x), 6))


# --------------------------------------------------------------------------- classification

def classification_metrics(y_true, y_pred, y_proba=None, prefix: str = "") -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        "accuracy": M.accuracy_score(y_true, y_pred),
        "balanced_accuracy": M.balanced_accuracy_score(y_true, y_pred),
        "precision_macro": M.precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": M.recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": M.f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": M.f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": M.matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": M.cohen_kappa_score(y_true, y_pred),
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        labels = np.arange(y_proba.shape[1])
        out["log_loss"] = M.log_loss(y_true, y_proba, labels=labels)
        try:
            if y_proba.shape[1] == 2:
                out["roc_auc"] = M.roc_auc_score(y_true, y_proba[:, 1])
                out["pr_auc"] = M.average_precision_score(y_true, y_proba[:, 1])
                out["brier"] = M.brier_score_loss(y_true, y_proba[:, 1])
            else:
                out["roc_auc"] = M.roc_auc_score(y_true, y_proba, multi_class="ovr", labels=labels)
        except ValueError:  # a split can be missing a class
            pass
    return {f"{prefix}{k}": _r(v) for k, v in out.items()}


def expected_calibration_error(y_true, prob_pos, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(prob_pos, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if mask.any():
            ece += mask.mean() * abs(np.mean(y_true[mask]) - np.mean(prob_pos[mask]))
    return float(ece)


def classification_report_payload(y_true, y_pred, y_proba, class_names: list[str]) -> dict[str, Any]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n_classes = len(class_names) or int(max(y_true.max(), y_pred.max()) + 1)
    class_names = class_names or [str(i) for i in range(n_classes)]
    labels = list(range(n_classes))
    cm = M.confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = M.precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    payload: dict[str, Any] = {
        "type": "classification",
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "per_class": [{"class": c, "precision": _r(p[i]), "recall": _r(r[i]), "f1": _r(f[i]), "support": int(s[i])}
                      for i, c in enumerate(class_names)],
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        roc, pr = [], []
        if n_classes == 2:
            targets = [(class_names[1], (y_true == 1).astype(int), y_proba[:, 1])]
        else:
            Y = label_binarize(y_true, classes=labels)
            targets = [(class_names[i], Y[:, i], y_proba[:, i]) for i in labels]
        for name, yt, ps in targets:
            if yt.min() == yt.max():
                continue
            fpr, tpr, _ = M.roc_curve(yt, ps)
            prec, rec, _ = M.precision_recall_curve(yt, ps)
            fx, ty = _downsample(fpr, tpr)
            rx, py = _downsample(rec[::-1], prec[::-1])
            roc.append({"label": name, "fpr": fx, "tpr": ty, "auc": _r(M.auc(fpr, tpr))})
            pr.append({"label": name, "recall": rx, "precision": py, "ap": _r(M.average_precision_score(yt, ps))})
        payload["roc_curves"], payload["pr_curves"] = roc, pr
        if n_classes == 2:
            frac_pos, mean_pred = calibration_curve(y_true, y_proba[:, 1], n_bins=10, strategy="quantile")
            payload["calibration"] = {"mean_predicted": np.round(mean_pred, 4).tolist(),
                                      "fraction_positive": np.round(frac_pos, 4).tolist(),
                                      "ece": _r(expected_calibration_error(y_true, y_proba[:, 1]))}
            # threshold sweep: helps pick an operating point other than 0.5
            ths = np.linspace(0.05, 0.95, 19)
            payload["thresholds"] = [{"threshold": _r(t),
                                      "precision": _r(M.precision_score(y_true, y_proba[:, 1] >= t, zero_division=0)),
                                      "recall": _r(M.recall_score(y_true, y_proba[:, 1] >= t, zero_division=0)),
                                      "f1": _r(M.f1_score(y_true, y_proba[:, 1] >= t, zero_division=0))} for t in ths]
    return payload


# --------------------------------------------------------------------------- regression

def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mse = M.mean_squared_error(y_true, y_pred)
    out = {"mae": M.mean_absolute_error(y_true, y_pred), "mse": mse, "rmse": np.sqrt(mse),
           "r2": M.r2_score(y_true, y_pred), "explained_variance": M.explained_variance_score(y_true, y_pred),
           "median_ae": M.median_absolute_error(y_true, y_pred)}
    nonzero = np.abs(y_true) > 1e-8
    if nonzero.any():
        out["mape"] = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])))
    return {f"{prefix}{k}": _r(v) for k, v in out.items()}


def regression_report_payload(y_true, y_pred) -> dict[str, Any]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    residuals = y_true - y_pred
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_true), size=min(400, len(y_true)), replace=False)
    counts, edges = np.histogram(residuals, bins=25)
    return {"type": "regression",
            "scatter": {"actual": np.round(y_true[idx], 4).tolist(), "predicted": np.round(y_pred[idx], 4).tolist()},
            "residual_hist": {"counts": counts.tolist(), "edges": np.round(edges, 4).tolist()},
            "residual_mean": _r(residuals.mean()), "residual_std": _r(residuals.std())}


# --------------------------------------------------------------------------- model-agnostic

def feature_importance(model, X, y, feature_names: list[str], use_permutation: bool = True,
                       top_k: int = 20, seed: int = 0) -> dict[str, Any] | None:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    method, values = None, None
    if hasattr(est, "feature_importances_"):
        method, values = "impurity", np.asarray(est.feature_importances_)
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        method, values = "|coefficient|", np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    elif use_permutation:
        res = permutation_importance(model, X, y, n_repeats=5, random_state=seed, n_jobs=1)
        method, values = "permutation", res.importances_mean
    if values is None or len(values) != len(feature_names):
        return None
    order = np.argsort(values)[::-1][:top_k]
    return {"method": method, "features": [feature_names[i] for i in order],
            "importance": [_r(values[i]) for i in order]}


def inference_profile(model, X, repeats: int = 5) -> dict[str, float]:
    """Latency per sample (batch predict) and serialized model size — the numbers serving cares about."""
    sample = X[: min(len(X), 256)]
    model.predict(sample)  # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.predict(sample)
    per_sample_ms = (time.perf_counter() - t0) / (repeats * len(sample)) * 1000
    t1 = time.perf_counter()
    model.predict(X[:1])
    single_ms = (time.perf_counter() - t1) * 1000
    return {"latency_ms_per_sample": _r(per_sample_ms), "latency_ms_single": _r(single_ms),
            "model_size_kb": _r(len(pickle.dumps(model)) / 1024)}
