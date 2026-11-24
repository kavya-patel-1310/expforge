"""ExpForge — a lightweight ML experimentation platform.

Track experiments, run hyperparameter sweeps, evaluate and compare models,
and promote the best ones through a model registry with a built-in prediction API.
"""
__version__ = "1.0.0"

from expforge.sdk import Client, HttpTransport, Run, init  # noqa: E402

__all__ = ["Client", "HttpTransport", "Run", "init", "__version__"]
