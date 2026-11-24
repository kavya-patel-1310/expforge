"""Programmatic tour: train from a config, compare runs, register, promote, and call the prediction API.

    expforge server &
    python examples/client_api_tour.py
"""
from expforge import Client
from expforge.training import run_from_config

client = Client("http://127.0.0.1:8000")

base = {"experiment": "api-tour", "dataset": {"name": "wine", "seed": 3}, "training": {"cv_folds": 5}}
ids = []
for model in ("logistic_regression", "random_forest", "knn"):
    result = run_from_config({**base, "name": model, "model": {"name": model}}, client)
    ids += result["run_ids"]

cmp = client.compare(ids)
best_id = cmp["best_run_per_metric"]["val_f1_macro"]
print("val_f1_macro per run:", dict(zip(ids, cmp["metrics"]["val_f1_macro"], strict=True)))
print("best run:", best_id)

model = client.register_model("wine-api-tour", best_id)
version = model["versions"][-1]["version"]
client.set_stage("wine-api-tour", version, "Production")

sample = [[13.2, 1.8, 2.4, 16.0, 100.0, 2.6, 2.7, 0.3, 1.8, 5.0, 1.0, 3.1, 1050.0]]
print(client.predict("wine-api-tour", sample, return_proba=True))
