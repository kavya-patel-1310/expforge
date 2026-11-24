"""Hyperparameter sweeps: grid search and random search over dotted config paths.

search_space entries:
    model.params.n_estimators: [100, 200, 400]                       # choice (grid or random)
    model.params.max_depth:    {type: int, low: 2, high: 16}         # random only
    model.params.learning_rate: {type: loguniform, low: 0.001, high: 0.3}
    model.params.subsample:    {type: uniform, low: 0.5, high: 1.0}
"""
from __future__ import annotations

import copy
import itertools
import logging
import math
from collections.abc import Iterator
from typing import Any

import numpy as np

from expforge.sdk import Client, ExpForgeError
from expforge.training.config import set_dotted
from expforge.training.datasets import load_dataset
from expforge.training.trainer import run_single

log = logging.getLogger("expforge.sweep")


def sample_value(spec: Any, rng: np.random.Generator) -> Any:
    if isinstance(spec, list):
        return spec[int(rng.integers(len(spec)))]
    kind = spec.get("type", "uniform")
    low, high = spec.get("low"), spec.get("high")
    if kind == "int":
        return int(rng.integers(low, high + 1))
    if kind == "uniform":
        return float(rng.uniform(low, high))
    if kind == "loguniform":
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    if kind == "choice":
        vals = spec["values"]
        return vals[int(rng.integers(len(vals)))]
    raise ValueError(f"unknown search-space type {kind!r}")


def generate_trials(search_space: dict[str, Any], strategy: str, n_trials: int, seed: int = 0) -> Iterator[dict]:
    keys = list(search_space)
    if strategy == "grid":
        for combo in itertools.product(*(search_space[k] for k in keys)):
            yield dict(zip(keys, combo, strict=True))
        return
    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    attempts = 0
    while len(seen) < n_trials and attempts < n_trials * 20:
        attempts += 1
        trial = {k: sample_value(search_space[k], rng) for k in keys}
        sig = repr(sorted(trial.items()))
        if sig not in seen:  # avoid re-running identical configs
            seen.add(sig)
            yield trial


def run_sweep(cfg: dict[str, Any], client: Client) -> dict[str, Any]:
    sw = cfg["sweep"]
    name = sw.get("name") or f"{cfg['model']['name']}-{sw['strategy']}"
    sweep = client.create_sweep(cfg["experiment"], name, sw["strategy"], sw["objective"], sw["direction"],
                                sw["search_space"])
    sweep_id = sweep["id"]
    maximize = sw["direction"] == "maximize"
    ds_cfg = cfg["dataset"]
    dataset = load_dataset(ds_cfg["name"], **ds_cfg.get("params", {}))  # load once, share across trials

    run_ids, best_id, best_val = [], None, None
    for i, trial in enumerate(generate_trials(sw["search_space"], sw["strategy"], int(sw["n_trials"]),
                                              int(sw.get("seed", 0))), 1):
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg.pop("sweep", None)
        trial_cfg.pop("register", None)
        for path, value in trial.items():
            set_dotted(trial_cfg, path, value)
        trial_cfg.setdefault("tags", {})["trial"] = i
        try:
            rid = run_single(trial_cfg, client, sweep_id=sweep_id, run_name=f"{name}-t{i:02d}", dataset=dataset)
        except (ValueError, ExpForgeError) as exc:  # bad combo (e.g. invalid param pair): record and continue
            log.warning("trial %d failed: %s", i, exc)
            continue
        run_ids.append(rid)
        score = client.get_run(rid)["summary"].get(sw["objective"])
        if isinstance(score, (int, float)) and (best_val is None or (score > best_val if maximize else score < best_val)):
            best_val, best_id = score, rid
            client.update_sweep(sweep_id, best_run_id=best_id)
        log.info("trial %d -> %s=%s (best=%s)", i, sw["objective"], score, best_val)

    client.update_sweep(sweep_id, status="finished", best_run_id=best_id)
    return {"run_ids": run_ids, "sweep_id": sweep_id, "best_run_id": best_id, "best_value": best_val}
