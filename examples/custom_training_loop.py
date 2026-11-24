"""Track ANY training loop with the SDK — here, logistic regression written from scratch in NumPy.

    expforge server &          # in another terminal
    python examples/custom_training_loop.py

The same three calls (start_run / log / log_summary) work for PyTorch, TensorFlow, XGBoost, etc.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import expforge
from expforge.evaluation import classification_metrics, classification_report_payload


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bce(X, y, w, b):
    p = np.clip(sigmoid(X @ w + b), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main(lr: float = 0.1, epochs: int = 200, l2: float = 1e-3, batch_size: int = 64):
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_val = scaler.transform(X_tr), scaler.transform(X_val)
    rng = np.random.default_rng(0)
    w, b = np.zeros(X.shape[1]), 0.0

    params = {"model": "numpy_logreg", "lr": lr, "epochs": epochs, "l2": l2, "batch_size": batch_size}
    with expforge.init(experiment="from-scratch", name=f"numpy-logreg-lr{lr}", params=params,
                       tags={"framework": "numpy"}) as run:
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(X_tr))
            for i in range(0, len(order), batch_size):
                idx = order[i:i + batch_size]
                p = sigmoid(X_tr[idx] @ w + b)
                grad = p - y_tr[idx]
                w -= lr * (X_tr[idx].T @ grad / len(idx) + l2 * w)
                b -= lr * grad.mean()

            if epoch % 5 == 0:
                val_acc = float(((sigmoid(X_val @ w + b) > 0.5) == y_val).mean())
                run.log({"train_loss": bce(X_tr, y_tr, w, b), "val_loss": bce(X_val, y_val, w, b), "val_accuracy": val_acc},
                        step=epoch)

        proba1 = sigmoid(X_val @ w + b)
        proba = np.c_[1 - proba1, proba1]
        pred = (proba1 > 0.5).astype(int)
        run.log_summary(classification_metrics(y_val, pred, proba, prefix="test_"))
        run.log_evaluation(classification_report_payload(y_val, pred, proba, ["malignant", "benign"]))
        print(f"logged run {run.id}: open http://127.0.0.1:8000/ui/#/run/{run.id}")


if __name__ == "__main__":
    for lr in (0.3, 0.05, 0.01):
        main(lr=lr)
