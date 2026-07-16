"""Retraining zone.

Catches the drift signal and runs the automated recovery: train a fresh model on
the newly-collected data, evaluate it, register it as a new version in the model
registry, and hand it back for hot-reload into the serving zone. In production the
GitHub Actions ``retrain.yml`` workflow performs these same steps on a
``repository_dispatch`` from the drift webhook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import Settings
from .logging_setup import get_logger
from .model.gbdt import build_model
from .model.metrics import classification_metrics
from .models import Dataset, ModelVersion
from .registry import LocalRegistry

log = get_logger("retraining")


@dataclass
class RetrainResult:
    version: ModelVersion
    model: object
    improved: bool
    prev_f1: float
    new_f1: float


class RetrainOrchestrator:
    def __init__(self, registry: LocalRegistry, settings: Settings) -> None:
        self.registry = registry
        self.settings = settings

    def retrain(self, train: Dataset, test: Dataset, reason: str = "drift",
                prev_f1: float = 0.0) -> RetrainResult:
        s = self.settings
        model_params = {"n_estimators": s.n_estimators, "max_depth": s.max_depth,
                        "learning_rate": s.learning_rate, "reg_lambda": s.reg_lambda,
                        "min_child_weight": s.min_child_weight, "max_bins": s.max_bins}
        params = {"backend": s.model_backend, **model_params}
        model = build_model(s.model_backend, **model_params)
        model.fit(train.X, train.y)
        proba = model.predict_proba(test.X)
        metrics = classification_metrics(test.y, proba)
        mv = self.registry.log_model(model, metrics, params, reason=reason)
        improved = metrics["f1"] >= prev_f1
        log.info("retrained v%d (%s): F1 %.3f (was %.3f)", mv.version, reason, metrics["f1"], prev_f1)
        return RetrainResult(version=mv, model=model, improved=improved,
                             prev_f1=prev_f1, new_f1=metrics["f1"])
