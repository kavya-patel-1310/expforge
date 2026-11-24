"""Command-line interface.

    expforge server  [--host 0.0.0.0] [--port 8000]     start API + dashboard
    expforge train   configs/rf_breast_cancer.yaml      run one config (single run or sweep)
    expforge demo                                       seed the store with realistic experiments
    expforge runs    [--experiment-id N]                list runs in the terminal
    expforge catalog                                    list built-in datasets and models
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from expforge import __version__


def _client(url: str | None):
    from expforge.sdk import Client
    return Client(url)


def cmd_server(args):
    import uvicorn

    from expforge.sdk import HttpTransport
    from expforge.server.app import create_app
    self_url = f"http://127.0.0.1:{args.port}"
    app = create_app(transport_factory=lambda: HttpTransport(self_url))
    print(f"ExpForge {__version__}  ->  dashboard http://{args.host}:{args.port}/ui   api docs http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_train(args):
    from expforge.training.config import load_config
    from expforge.training.trainer import run_from_config
    result = run_from_config(load_config(args.config), _client(args.url))
    print(json.dumps(result, indent=2))


def _local_client():
    """In-process client bound directly to the local store (no server needed)."""
    from fastapi.testclient import TestClient

    from expforge.sdk import Client, HttpTransport
    from expforge.server.app import create_app
    app = create_app()
    tc = TestClient(app)
    return Client(transport=HttpTransport("", session=tc))


def cmd_demo(args):
    from expforge.training.config import load_config
    from expforge.training.trainer import run_from_config
    client = _local_client() if args.url is None else _client(args.url)
    candidates = [Path.cwd() / "configs", Path(__file__).resolve().parent.parent / "configs"]
    cfg_dir = next((c for c in candidates if c.is_dir()), candidates[0])
    configs = sorted(cfg_dir.glob("*.yaml"))
    if not configs:
        sys.exit("no configs/ directory found; run from the repo root")
    for path in configs:
        if args.quick and "sweep" in path.name:
            continue
        print(f"→ {path.name}")
        result = run_from_config(load_config(path), client)
        print(f"   runs={result['run_ids']} best={result.get('best_run_id')}"
              + (f" registered={result['registered']}" if result.get("registered") else ""))
    print("\nDone. Start the dashboard with:  expforge server")


def cmd_runs(args):
    client = _local_client() if args.url is None else _client(args.url)
    runs = client.runs(args.experiment_id)
    print(f"{'ID':>4}  {'EXPERIMENT':<22} {'NAME':<26} {'STATUS':<9} {'MODEL':<22} KEY METRIC")
    for r in runs:
        s = r["summary"]
        key = next((k for k in ("test_f1_macro", "test_r2", "test_accuracy") if k in s), None)
        metric = f"{key}={s[key]:.4f}" if key else ""
        print(f"{r['id']:>4}  {r['experiment_name'][:22]:<22} {r['name'][:26]:<26} {r['status']:<9} "
              f"{str(r['params'].get('model', ''))[:22]:<22} {metric}")


def cmd_catalog(_args):
    from expforge.training.datasets import DATASETS
    from expforge.training.models import MODELS
    print("Datasets:")
    for k, v in DATASETS.items():
        print(f"  {k:<26} [{v.task}] {v.description}")
    print("\nModels:")
    for k, v in MODELS.items():
        print(f"  {k:<26} tasks={','.join(sorted(v.tasks))}  curves={v.iterative or '-'}")


def main(argv: list[str] | None = None):
    logging.basicConfig(level=os.environ.get("EXPFORGE_LOG", "WARNING"), format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="expforge", description="ML experimentation platform")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("server", help="start the tracking server + dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_server)

    t = sub.add_parser("train", help="train from a YAML config")
    t.add_argument("config")
    t.add_argument("--url", default=None, help="server URL (default $EXPFORGE_URL or http://127.0.0.1:8000)")
    t.set_defaults(fn=cmd_train)

    d = sub.add_parser("demo", help="populate the store with example experiments")
    d.add_argument("--url", default=None, help="log to a running server instead of the local store")
    d.add_argument("--quick", action="store_true", help="skip sweeps")
    d.set_defaults(fn=cmd_demo)

    r = sub.add_parser("runs", help="list runs")
    r.add_argument("--experiment-id", type=int, default=None)
    r.add_argument("--url", default=None)
    r.set_defaults(fn=cmd_runs)

    c = sub.add_parser("catalog", help="list datasets and models")
    c.set_defaults(fn=cmd_catalog)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
