"""Training config: loading, validation and defaults.

Example (YAML):

    experiment: breast-cancer
    dataset: {name: breast_cancer, test_size: 0.2, val_size: 0.2, seed: 42}
    model:   {name: random_forest, params: {n_estimators: 300}}
    training: {scale: true, cv_folds: 5, epochs: 60, early_stopping_patience: 8}
    sweep:   {strategy: random, n_trials: 20, objective: val_f1_macro, direction: maximize,
              search_space: {model.params.max_depth: {type: int, low: 2, high: 16}}}
    register: {name: cancer-classifier, stage: Staging}
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from expforge.training.datasets import DATASETS
from expforge.training.models import MODELS

DEFAULT_TRAINING = {"scale": True, "cv_folds": 0, "epochs": 50, "early_stopping_patience": 8,
                    "log_every": 1, "permutation_importance": True}
DEFAULT_DATASET = {"test_size": 0.2, "val_size": 0.2, "seed": 42, "params": {}}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return validate_config(yaml.safe_load(fh))


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    cfg = copy.deepcopy(raw)
    for key in ("experiment", "dataset", "model"):
        if key not in cfg:
            raise ValueError(f"config missing required key: {key!r}")

    if isinstance(cfg["dataset"], str):
        cfg["dataset"] = {"name": cfg["dataset"]}
    if isinstance(cfg["model"], str):
        cfg["model"] = {"name": cfg["model"]}
    cfg["dataset"] = {**DEFAULT_DATASET, **cfg["dataset"]}
    cfg["model"].setdefault("params", {})
    cfg["training"] = {**DEFAULT_TRAINING, **(cfg.get("training") or {})}
    cfg.setdefault("tags", {})

    ds_name, model_name = cfg["dataset"]["name"], cfg["model"]["name"]
    if ds_name not in DATASETS:
        raise ValueError(f"unknown dataset {ds_name!r}; available: {sorted(DATASETS)}")
    if model_name not in MODELS:
        raise ValueError(f"unknown model {model_name!r}; available: {sorted(MODELS)}")
    task = DATASETS[ds_name].task
    if task == "any":
        task = cfg["dataset"]["params"].get("task", "classification")
    if task not in MODELS[model_name].tasks:
        raise ValueError(f"model {model_name!r} cannot be used for {task}")
    cfg["task"] = task

    ds = cfg["dataset"]
    if not 0 < ds["test_size"] < 0.9 or not 0 < ds["val_size"] < 0.9 or ds["test_size"] + ds["val_size"] >= 0.95:
        raise ValueError("test_size / val_size must leave data for training")

    sweep = cfg.get("sweep")
    if sweep:
        sweep.setdefault("strategy", "random")
        sweep.setdefault("direction", "maximize")
        sweep.setdefault("n_trials", 10)
        if sweep["strategy"] not in ("grid", "random"):
            raise ValueError("sweep.strategy must be 'grid' or 'random'")
        if not sweep.get("objective"):
            raise ValueError("sweep.objective is required (e.g. val_f1_macro, val_rmse)")
        if not sweep.get("search_space"):
            raise ValueError("sweep.search_space is required")
        if sweep["strategy"] == "grid" and not all(isinstance(v, list) for v in sweep["search_space"].values()):
            raise ValueError("grid search requires every search_space value to be a list")
    return cfg


def set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    node = cfg
    *parents, leaf = dotted.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value
