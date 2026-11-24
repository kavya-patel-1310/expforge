"""Model zoo: a uniform factory over scikit-learn estimators for classification and regression.

`iterative` marks how a model exposes training progress so the trainer can log learning curves:
  * "partial_fit" -> trained epoch-by-epoch (MLP, SGD) with early stopping
  * "staged"      -> boosting; curves recovered from staged predictions
  * None          -> single fit (optionally k-fold cross-validated)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sklearn import ensemble, linear_model, neighbors, neural_network, svm, tree
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class ModelSpec:
    tasks: set[str]
    build: Callable[[str, dict[str, Any]], Any]
    defaults: dict[str, Any] = field(default_factory=dict)
    iterative: str | None = None


def _pick(clf, reg):
    def build(task: str, params: dict[str, Any]):
        cls = clf if task == "classification" else reg
        if cls is None:
            raise ValueError(f"model does not support task {task!r}")
        return cls(**params)
    return build


def _sgd(task: str, params: dict[str, Any]):
    params = dict(params)
    if task == "classification":
        params.setdefault("loss", "log_loss")  # gives predict_proba
        return linear_model.SGDClassifier(**params)
    params.setdefault("loss", "squared_error")
    return linear_model.SGDRegressor(**params)


def _svm(task: str, params: dict[str, Any]):
    if task == "classification":  # Platt-scaled probabilities (replaces deprecated SVC(probability=True))
        return CalibratedClassifierCV(svm.SVC(**params), ensemble=False, cv=3)
    return svm.SVR(**params)


MODELS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec({"classification"}, _pick(linear_model.LogisticRegression, None),
                                     {"C": 1.0, "max_iter": 2000}),
    "ridge": ModelSpec({"regression"}, _pick(None, linear_model.Ridge), {"alpha": 1.0}),
    "decision_tree": ModelSpec({"classification", "regression"},
                               _pick(tree.DecisionTreeClassifier, tree.DecisionTreeRegressor),
                               {"max_depth": 6, "random_state": 0}),
    "random_forest": ModelSpec({"classification", "regression"},
                               _pick(ensemble.RandomForestClassifier, ensemble.RandomForestRegressor),
                               {"n_estimators": 200, "max_depth": None, "n_jobs": -1, "random_state": 0}),
    "extra_trees": ModelSpec({"classification", "regression"},
                             _pick(ensemble.ExtraTreesClassifier, ensemble.ExtraTreesRegressor),
                             {"n_estimators": 300, "n_jobs": -1, "random_state": 0}),
    "gradient_boosting": ModelSpec({"classification", "regression"},
                                   _pick(ensemble.GradientBoostingClassifier, ensemble.GradientBoostingRegressor),
                                   {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3, "random_state": 0},
                                   iterative="staged"),
    "hist_gradient_boosting": ModelSpec({"classification", "regression"},
                                        _pick(ensemble.HistGradientBoostingClassifier,
                                              ensemble.HistGradientBoostingRegressor),
                                        {"max_iter": 200, "learning_rate": 0.1, "random_state": 0}),
    "svm": ModelSpec({"classification", "regression"}, _svm, {"C": 1.0, "kernel": "rbf"}),
    "knn": ModelSpec({"classification", "regression"},
                     _pick(neighbors.KNeighborsClassifier, neighbors.KNeighborsRegressor), {"n_neighbors": 7}),
    "mlp": ModelSpec({"classification", "regression"},
                     _pick(neural_network.MLPClassifier, neural_network.MLPRegressor),
                     {"hidden_layer_sizes": [64, 32], "learning_rate_init": 0.001, "alpha": 1e-4, "random_state": 0},
                     iterative="partial_fit"),
    "sgd": ModelSpec({"classification", "regression"}, _sgd,
                     {"alpha": 1e-4, "learning_rate": "optimal", "random_state": 0}, iterative="partial_fit"),
}


def build_model(name: str, task: str, params: dict[str, Any] | None = None):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    spec = MODELS[name]
    if task not in spec.tasks:
        raise ValueError(f"model {name!r} does not support {task}")
    merged = {**spec.defaults, **(params or {})}
    if "hidden_layer_sizes" in merged:
        merged["hidden_layer_sizes"] = tuple(merged["hidden_layer_sizes"])
    return spec.build(task, merged), merged
