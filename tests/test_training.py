import copy
import time

import pytest
from conftest import FAST_CLF

from expforge.training.config import validate_config
from expforge.training.sweep import generate_trials
from expforge.training.trainer import run_from_config, run_single


@pytest.mark.parametrize("bad, msg", [
    ({"dataset": "iris", "model": "ridge"}, "missing required key"),
    ({"experiment": "e", "dataset": "nope", "model": "ridge"}, "unknown dataset"),
    ({"experiment": "e", "dataset": "iris", "model": "ridge"}, "cannot be used for classification"),
    ({"experiment": "e", "dataset": "iris", "model": "knn", "sweep": {"objective": "x", "strategy": "grid",
      "search_space": {"model.params.n_neighbors": {"type": "int", "low": 1, "high": 3}}}}, "grid search requires"),
])
def test_config_validation(bad, msg):
    with pytest.raises(ValueError, match=msg):
        validate_config(bad)


def test_single_run_end_to_end(client):
    rid = run_single(FAST_CLF, client)
    run = client.get_run(rid)
    s, ev = run["summary"], client._t.request("GET", f"/api/runs/{rid}")["evaluation"]
    assert run["status"] == "finished"
    assert s["test_accuracy"] > 0.9 and s["cv_folds"] == 3 and "latency_ms_per_sample" in s
    assert ev["type"] == "classification" and len(ev["confusion_matrix"]) == 2 and ev["roc_curves"]
    assert ev["feature_importance"]["method"] == "|coefficient|"
    assert run["system_info"]["dataset_fingerprint"]  # data lineage captured
    assert any(a["kind"] == "model" for a in run["artifacts"])


def test_iterative_model_logs_curves_with_early_stopping(client):
    cfg = {**FAST_CLF, "model": {"name": "sgd"}, "training": {"epochs": 40, "early_stopping_patience": 3}}
    rid = run_single(cfg, client)
    series = client._t.request("GET", f"/api/runs/{rid}/metrics")
    assert {"train_loss", "val_loss", "train_accuracy", "val_accuracy"} <= series.keys()
    s = client.get_run(rid)["summary"]
    assert 1 <= s["best_epoch"] <= s["epochs_trained"] <= 40


def test_staged_boosting_regression(client):
    cfg = {"experiment": "t-reg", "dataset": {"name": "diabetes"},
           "model": {"name": "gradient_boosting", "params": {"n_estimators": 30}}, "training": {"scale": False}}
    rid = run_single(cfg, client)
    series = client._t.request("GET", f"/api/runs/{rid}/metrics")
    assert series["val_loss"][-1]["step"] == 30 and "val_r2" in series
    assert "test_rmse" in client.get_run(rid)["summary"]


def test_reproducible_given_seed(client):
    a, b = run_single(FAST_CLF, client), run_single(FAST_CLF, client)
    sa, sb = client.get_run(a)["summary"], client.get_run(b)["summary"]
    assert sa["test_f1_macro"] == sb["test_f1_macro"]


def test_trial_generation():
    grid = list(generate_trials({"a": [1, 2], "b": ["x", "y", "z"]}, "grid", 0))
    assert len(grid) == 6 and {"a": 2, "b": "z"} in grid
    rand = list(generate_trials({"lr": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
                                 "d": {"type": "int", "low": 2, "high": 4}}, "random", 15, seed=1))
    assert len(rand) == 15 and all(1e-4 <= t["lr"] <= 1e-1 and 2 <= t["d"] <= 4 for t in rand)
    assert len(list(generate_trials({"a": [1, 2]}, "random", 10))) == 2  # dedup caps at unique configs


def test_sweep_registers_best_and_serves_predictions(client):
    cfg = copy.deepcopy(FAST_CLF)
    cfg["training"] = {"cv_folds": 0}
    cfg["sweep"] = {"strategy": "grid", "objective": "val_f1_macro", "search_space": {"model.params.C": [0.01, 1.0, 10.0]}}
    cfg["register"] = {"name": "clf", "stage": "Production"}
    result = run_from_config(cfg, client)
    assert len(result["run_ids"]) == 3 and result["best_run_id"] in result["run_ids"]
    sweep = client._t.request("GET", "/api/sweeps")[0]
    assert sweep["status"] == "finished" and sweep["best_run_id"] == result["best_run_id"]
    best_val = client.get_run(result["best_run_id"])["summary"]["val_f1_macro"]
    assert all(client.get_run(r)["summary"]["val_f1_macro"] <= best_val for r in result["run_ids"])

    pred = client.predict("clf", [[0.0] * 30, [1.0] * 30], return_proba=True)
    assert pred["version"] == 1 and len(pred["predictions"]) == 2 and len(pred["probabilities"][0]) == 2


def test_registry_stage_transitions(client):
    r1, r2 = run_single(FAST_CLF, client), run_single(FAST_CLF, client)
    client.register_model("m", r1)
    client.register_model("m", r2)
    client.set_stage("m", 1, "Production")
    client.set_stage("m", 2, "Production")  # v1 should be auto-archived
    versions = {v["version"]: v["stage"] for v in client._t.request("GET", "/api/models")[0]["versions"]}
    assert versions == {1: "Archived", 2: "Production"}
    assert client.predict("m", [[0.0] * 30])["version"] == 2


def test_predict_errors(server, client):
    rid = run_single(FAST_CLF, client)
    client.register_model("m2", rid)
    assert server.post("/api/models/m2/predict", json={"instances": [[0.0] * 30]}).status_code == 409  # none in prod
    assert server.post("/api/models/m2/predict", json={"instances": [[0.0] * 3], "version": 1}).status_code == 422
    assert server.post("/api/models/nope/predict", json={"instances": [[0.0]]}).status_code == 404


def test_background_job(server):
    job = server.post("/api/jobs", json={"config": FAST_CLF}).json()
    for _ in range(100):
        job = server.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("finished", "failed"):
            break
        time.sleep(0.2)
    assert job["status"] == "finished", job["error"]
    assert len(job["run_ids"]) == 1
    assert server.post("/api/jobs", json={"config": {"experiment": "x"}}).status_code == 422
