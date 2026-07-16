"""DriftGuard — real-time streaming & drift-detection MLOps pipeline.

A self-monitoring fraud-detection system: it serves predictions from a trained
model, streams live transactions, continuously compares the live distribution
against the training reference with the Population Stability Index, and — when the
data drifts past threshold — fires an alert and automatically retrains, registers,
and hot-reloads a new model version.

Offline-first: a from-scratch XGBoost-style gradient-boosted tree, a local
MLflow-style registry, PSI drift detection, and an in-memory Kafka-style bus mean
the whole four-zone pipeline runs, tests and verifies with zero external services.
Real adapters (XGBoost, MLflow, Kafka, Evidently) wire in via configuration.
"""

from .version import __version__
from .config import Settings
from .engine import DriftGuard

__all__ = ["__version__", "Settings", "DriftGuard"]
