import hashlib


def _run(server, **kw):
    body = {"experiment": "exp-a", "params": {"lr": 0.1}, **kw}
    r = server.post("/api/runs", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_health_and_empty_stats(server):
    assert server.get("/api/health").json()["status"] == "ok"
    stats = server.get("/api/stats").json()
    assert stats["runs"] == 0 and stats["experiments"] == 0


def test_run_creates_experiment_and_autonames(server):
    r1, r2 = _run(server), _run(server)
    assert r1["experiment_name"] == "exp-a" and r1["name"] == "run-1" and r2["name"] == "run-2"
    exps = server.get("/api/experiments").json()
    assert len(exps) == 1 and exps[0]["run_count"] == 2


def test_duplicate_experiment_conflict(server):
    assert server.post("/api/experiments", json={"name": "x"}).status_code == 200
    assert server.post("/api/experiments", json={"name": "x"}).status_code == 409


def test_patch_merges_json_and_sets_end_time(server):
    run = _run(server)
    server.patch(f"/api/runs/{run['id']}", json={"summary": {"acc": 0.9}})
    out = server.patch(f"/api/runs/{run['id']}", json={"summary": {"f1": 0.8}, "params": {"bs": 32}, "status": "finished"}).json()
    assert out["summary"] == {"acc": 0.9, "f1": 0.8}
    assert out["params"] == {"lr": 0.1, "bs": 32}
    assert out["ended_at"] is not None and out["duration_s"] is not None


def test_metrics_roundtrip(server):
    run = _run(server)
    pts = [{"key": "loss", "value": 1.0 / (i + 1), "step": i} for i in range(5)]
    assert server.post(f"/api/runs/{run['id']}/metrics", json={"metrics": pts}).json()["logged"] == 5
    series = server.get(f"/api/runs/{run['id']}/metrics").json()
    assert [p["step"] for p in series["loss"]] == [0, 1, 2, 3, 4]
    assert server.get(f"/api/runs/{run['id']}").json()["metric_keys"] == ["loss"]


def test_artifact_upload_download_checksum(server):
    run = _run(server)
    payload = b"hello model" * 100
    art = server.post(f"/api/runs/{run['id']}/artifacts", files={"file": ("../../evil/w.bin", payload)}, data={"kind": "model"}).json()
    assert art["name"] == "w.bin"  # path traversal stripped
    assert art["sha256"] == hashlib.sha256(payload).hexdigest() and art["size_bytes"] == len(payload)
    assert server.get(f"/api/artifacts/{art['id']}/download").content == payload


def test_compare_and_leaderboard(server):
    ids = []
    for lr, acc, loss in [(0.1, 0.80, 0.5), (0.01, 0.92, 0.3), (0.001, 0.85, 0.2)]:
        run = _run(server, params={"lr": lr, "model": "m"})
        server.patch(f"/api/runs/{run['id']}", json={"summary": {"val_acc": acc, "val_loss": loss}, "status": "finished"})
        ids.append(run["id"])
    cmp = server.post("/api/compare", json={"run_ids": ids}).json()
    assert cmp["differing_params"] == ["lr"]
    assert cmp["best_run_per_metric"] == {"val_acc": ids[1], "val_loss": ids[2]}  # loss -> lower is better
    lb = server.get(f"/api/experiments/{cmp['runs'][0]['experiment_id']}/leaderboard?metric=val_acc").json()
    assert [row["run_id"] for row in lb] == [ids[1], ids[2], ids[0]]


def test_filters_and_404(server):
    a = _run(server, tags={"team": "x"})
    _run(server, tags={"team": "y"})
    assert [r["id"] for r in server.get("/api/runs?tag=team=x").json()] == [a["id"]]
    assert server.get("/api/runs/9999").status_code == 404
    assert server.post("/api/compare", json={"run_ids": [a["id"], 9999]}).status_code == 404


def test_delete_experiment_cascades(server):
    run = _run(server)
    server.post(f"/api/runs/{run['id']}/metrics", json={"metrics": [{"key": "a", "value": 1}]})
    server.delete(f"/api/experiments/{run['experiment_id']}")
    assert server.get(f"/api/runs/{run['id']}").status_code == 404
    assert server.get("/api/stats").json()["metric_points"] == 0


def test_dashboard_served(server):
    r = server.get("/ui/")
    assert r.status_code == 200 and "ExpForge" in r.text
