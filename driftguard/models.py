"""Core domain models (dataclasses)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

FEATURES = [
    "amount", "hour", "distance_from_home", "txn_last_hour",
    "merchant_risk", "account_age_days", "foreign", "online",
]


@dataclass
class Dataset:
    X: list[list[float]]
    y: list[int]
    feature_names: list[str] = field(default_factory=lambda: list(FEATURES))

    def __len__(self) -> int:
        return len(self.X)

    def column(self, name: str) -> list[float]:
        idx = self.feature_names.index(name)
        return [row[idx] for row in self.X]

    @property
    def fraud_rate(self) -> float:
        return sum(self.y) / len(self.y) if self.y else 0.0


@dataclass
class Prediction:
    proba: float
    label: int
    model_version: int

    def to_dict(self) -> dict[str, Any]:
        return {"fraud_probability": round(self.proba, 4), "is_fraud": self.label,
                "model_version": self.model_version}


@dataclass
class ModelVersion:
    version: int
    metrics: dict[str, float]
    params: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    reason: str = "initial"

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
                "params": self.params, "reason": self.reason, "created_at": self.created_at}


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    drifted: bool

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "psi": round(self.psi, 4), "drifted": self.drifted}


@dataclass
class DriftReport:
    features: list[FeatureDrift]
    drift_share: float          # fraction of features drifted
    dataset_drift: bool         # pipeline-level alert
    n_samples: int
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"dataset_drift": self.dataset_drift, "drift_share": round(self.drift_share, 3),
                "n_samples": self.n_samples,
                "features": [f.to_dict() for f in self.features]}
