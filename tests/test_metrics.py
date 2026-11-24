import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, r2_score

from expforge.evaluation import metrics as E


def test_classification_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    proba1 = np.clip(y * 0.7 + rng.normal(0.15, 0.2, 300), 0.01, 0.99)
    proba = np.c_[1 - proba1, proba1]
    pred = proba.argmax(1)
    m = E.classification_metrics(y, pred, proba, prefix="val_")
    assert m["val_accuracy"] == pytest.approx(accuracy_score(y, pred), abs=1e-6)
    assert m["val_f1_macro"] == pytest.approx(f1_score(y, pred, average="macro"), abs=1e-6)
    assert {"val_roc_auc", "val_pr_auc", "val_brier", "val_log_loss", "val_mcc"} <= m.keys()


def test_report_payload_shapes_multiclass():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 200)
    proba = rng.dirichlet([1, 1, 1], 200)
    rep = E.classification_report_payload(y, proba.argmax(1), proba, ["a", "b", "c"])
    assert np.array(rep["confusion_matrix"]).shape == (3, 3)
    assert len(rep["roc_curves"]) == 3 and all(len(c["fpr"]) <= E.MAX_CURVE_POINTS for c in rep["roc_curves"])
    assert "calibration" not in rep  # binary only


def test_ece_perfect_vs_overconfident():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 20000)
    y_cal = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert E.expected_calibration_error(y_cal, p) < 0.02
    assert E.expected_calibration_error(y_cal, np.where(p > 0.5, 0.99, 0.01)) > 0.15


def test_regression_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.1, 1.9, 3.2, 3.8])
    m = E.regression_metrics(y, pred)
    assert m["r2"] == pytest.approx(r2_score(y, pred), abs=1e-6)
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean((y - pred) ** 2)), abs=1e-6)
    rep = E.regression_report_payload(y, pred)
    assert sum(rep["residual_hist"]["counts"]) == 4


def test_feature_importance_and_profile():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    X = np.random.default_rng(3).normal(size=(200, 4))
    y = (X[:, 2] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    fi = E.feature_importance(rf, X, y, ["a", "b", "c", "d"])
    assert fi["method"] == "impurity" and fi["features"][0] == "c"
    knn = KNeighborsClassifier().fit(X, y)
    assert E.feature_importance(knn, X, y, ["a", "b", "c", "d"])["method"] == "permutation"
    prof = E.inference_profile(rf, X)
    assert prof["latency_ms_per_sample"] > 0 and prof["model_size_kb"] > 0
