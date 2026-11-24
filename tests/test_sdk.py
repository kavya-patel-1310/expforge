import numpy as np
import pytest

from expforge.sdk import ExpForgeError, _jsonable


def test_context_manager_finishes_and_flushes(client):
    with client.start_run("sdk", params={"a": np.float32(0.5)}) as run:
        for step in range(10):
            run.log({"loss": 1 / (step + 1), "acc": step / 10}, step=step)
    data = client.get_run(run.id)
    assert data["status"] == "finished"
    assert data["params"]["a"] == pytest.approx(0.5)
    assert sorted(data["metric_keys"]) == ["acc", "loss"]
    assert "python" in data["system_info"]


def test_exception_marks_run_failed(client):
    with pytest.raises(RuntimeError):
        with client.start_run("sdk") as run:
            raise RuntimeError("boom")
    data = client.get_run(run.id)
    assert data["status"] == "failed" and "boom" in data["error"]


def test_nonfinite_metrics_dropped(client):
    with client.start_run("sdk") as run:
        run.log({"loss": float("nan"), "acc": 0.5, "x": float("inf")})
    assert client.get_run(run.id)["metric_keys"] == ["acc"]


def test_buffer_auto_flushes(client):
    run = client.start_run("sdk")
    run._flush_every = 5
    run.log({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
    assert run._buffer == []  # flushed without finish()
    run.finish()


def test_log_model_and_bytes(client):
    from sklearn.linear_model import LogisticRegression
    run = client.start_run("sdk")
    model = LogisticRegression().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    art = run.log_model(model)
    assert art["kind"] == "model" and art["size_bytes"] > 0
    run.log_bytes(b"{}", "notes.json")
    run.finish()
    assert {a["name"] for a in client.get_run(run.id)["artifacts"]} == {"model.joblib", "notes.json"}


def test_http_errors_raise(client):
    with pytest.raises(ExpForgeError):
        client.get_run(424242)


def test_jsonable():
    out = _jsonable({"a": np.int64(3), "b": (1, 2), "c": np.array([1.5]), "d": object})
    assert out["a"] == 3 and out["b"] == [1, 2] and out["c"] == [1.5] and isinstance(out["d"], str)
