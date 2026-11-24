"""Dataset registry. Every loader returns a `Dataset` with a content fingerprint for reproducibility."""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn import datasets as skd


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    task: str  # classification | regression
    feature_names: list[str]
    target_names: list[str] = field(default_factory=list)
    name: str = ""

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.X, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.y).astype(np.float64).tobytes())
        return h.hexdigest()[:16]

    def describe(self) -> dict[str, Any]:
        info = {"dataset_name": self.name, "dataset_fingerprint": self.fingerprint, "n_samples": int(self.X.shape[0]),
                "n_features": int(self.X.shape[1]), "task": self.task}
        if self.task == "classification":
            _, counts = np.unique(self.y, return_counts=True)
            info["n_classes"] = int(len(counts))
            info["class_balance"] = [round(c / counts.sum(), 4) for c in counts.tolist()]
        return info


@dataclass
class DatasetSpec:
    task: str
    description: str
    loader: Callable[..., Dataset]


def _from_sklearn(fn, name: str, task: str) -> Callable[..., Dataset]:
    def load(**_: Any) -> Dataset:
        b = fn()
        targets = [str(t) for t in getattr(b, "target_names", [])] if task == "classification" else []
        return Dataset(np.asarray(b.data, dtype=float), np.asarray(b.target), task,
                       [str(f) for f in getattr(b, "feature_names", [f"f{i}" for i in range(b.data.shape[1])])],
                       targets, name)
    return load


def _synthetic_classification(n_samples: int = 2000, n_features: int = 20, n_informative: int = 8,
                              n_classes: int = 2, class_sep: float = 1.0, flip_y: float = 0.03,
                              weights: list[float] | None = None, seed: int = 0, **_: Any) -> Dataset:
    X, y = skd.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                   n_redundant=max(0, min(4, n_features - n_informative)), n_classes=n_classes,
                                   class_sep=class_sep, flip_y=flip_y, weights=weights, random_state=seed)
    return Dataset(X, y, "classification", [f"x{i}" for i in range(n_features)],
                   [f"class_{i}" for i in range(n_classes)], "synthetic_classification")


def _synthetic_regression(n_samples: int = 2000, n_features: int = 15, n_informative: int = 6,
                          noise: float = 15.0, seed: int = 0, **_: Any) -> Dataset:
    X, y = skd.make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               noise=noise, random_state=seed)
    return Dataset(X, y, "regression", [f"x{i}" for i in range(n_features)], [], "synthetic_regression")


def _csv(path: str, target: str, task: str = "classification", **_: Any) -> Dataset:
    import pandas as pd
    df = pd.read_csv(path).dropna(subset=[target])
    y_raw = df.pop(target)
    X = pd.get_dummies(df, drop_first=True).astype(float)  # one-hot categoricals
    X = X.fillna(X.median(numeric_only=True))
    if task == "classification":
        codes, uniques = pd.factorize(y_raw, sort=True)
        return Dataset(X.to_numpy(), codes, task, list(X.columns), [str(u) for u in uniques], f"csv:{path}")
    return Dataset(X.to_numpy(), y_raw.to_numpy(dtype=float), task, list(X.columns), [], f"csv:{path}")


DATASETS: dict[str, DatasetSpec] = {
    "breast_cancer": DatasetSpec("classification", "Wisconsin breast cancer — binary, 569 x 30",
                                 _from_sklearn(skd.load_breast_cancer, "breast_cancer", "classification")),
    "wine": DatasetSpec("classification", "Wine cultivars — 3 classes, 178 x 13",
                        _from_sklearn(skd.load_wine, "wine", "classification")),
    "iris": DatasetSpec("classification", "Iris — 3 classes, 150 x 4",
                        _from_sklearn(skd.load_iris, "iris", "classification")),
    "digits": DatasetSpec("classification", "Handwritten digits 8x8 — 10 classes, 1797 x 64",
                          _from_sklearn(skd.load_digits, "digits", "classification")),
    "diabetes": DatasetSpec("regression", "Diabetes progression — regression, 442 x 10",
                            _from_sklearn(skd.load_diabetes, "diabetes", "regression")),
    "synthetic_classification": DatasetSpec("classification", "Configurable make_classification generator",
                                            _synthetic_classification),
    "synthetic_regression": DatasetSpec("regression", "Configurable make_regression generator",
                                        _synthetic_regression),
    "csv": DatasetSpec("any", "Your own CSV: params {path, target, task}", _csv),
}


def load_dataset(name: str, **params: Any) -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name].loader(**params)
