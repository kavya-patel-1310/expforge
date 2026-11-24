import pytest
from fastapi.testclient import TestClient

from expforge.sdk import Client, HttpTransport
from expforge.server.app import create_app


@pytest.fixture()
def server(tmp_path):
    """A fresh app + DB + artifact store per test. Background jobs talk back through the same TestClient."""
    holder = {}
    app = create_app(db_url=f"sqlite:///{tmp_path / 'test.db'}", artifact_root=tmp_path / "artifacts",
                     transport_factory=lambda: HttpTransport("", session=holder["tc"]))
    with TestClient(app) as tc:
        holder["tc"] = tc
        yield tc


@pytest.fixture()
def client(server) -> Client:
    return Client(transport=HttpTransport("", session=server))


FAST_CLF = {"experiment": "t-clf", "dataset": {"name": "breast_cancer", "seed": 0},
            "model": {"name": "logistic_regression"}, "training": {"cv_folds": 3}}
