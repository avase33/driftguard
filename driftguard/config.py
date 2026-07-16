"""Central configuration resolved from the environment.

Offline-first defaults keep all four zones in-process (from-scratch GBDT, local
registry, PSI drift detection, in-memory bus). Point adapters at XGBoost, MLflow,
Kafka and Evidently for production via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Model: gbdt | xgboost
    model_backend: str = "gbdt"
    n_estimators: int = 60
    max_depth: int = 3
    learning_rate: float = 0.3
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    max_bins: int = 32

    # Registry: local | mlflow
    registry_backend: str = "local"
    registry_dir: str = "mlruns_local"
    mlflow_uri: str = "http://localhost:5000"

    # Streaming: memory | kafka
    bus_backend: str = "memory"
    kafka_brokers: str = "localhost:9092"
    topic: str = "transactions"
    partitions: int = 4

    # Drift detection (PSI)
    psi_warn: float = 0.1
    psi_alert: float = 0.2            # PSI >= this = feature drifted
    drift_feature_fraction: float = 0.3   # >30% features drifted -> pipeline alert
    monitor_batch: int = 500         # rows batched before a drift check
    psi_bins: int = 10

    # Retraining
    auto_retrain: bool = True

    # Alerting
    webhook_url: str = ""
    slack_webhook: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        g = os.environ.get
        return cls(
            model_backend=g("DRIFTGUARD_MODEL", "gbdt"),
            n_estimators=int(g("DRIFTGUARD_ESTIMATORS", "60")),
            max_depth=int(g("DRIFTGUARD_MAX_DEPTH", "3")),
            learning_rate=float(g("DRIFTGUARD_LR", "0.3")),
            registry_backend=g("DRIFTGUARD_REGISTRY", "local"),
            registry_dir=g("DRIFTGUARD_REGISTRY_DIR", "mlruns_local"),
            mlflow_uri=g("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            bus_backend=g("DRIFTGUARD_BUS", "memory"),
            kafka_brokers=g("KAFKA_BROKERS", "localhost:9092"),
            topic=g("DRIFTGUARD_TOPIC", "transactions"),
            partitions=int(g("DRIFTGUARD_PARTITIONS", "4")),
            psi_alert=float(g("DRIFTGUARD_PSI_ALERT", "0.2")),
            drift_feature_fraction=float(g("DRIFTGUARD_DRIFT_FRACTION", "0.3")),
            monitor_batch=int(g("DRIFTGUARD_MONITOR_BATCH", "500")),
            auto_retrain=g("DRIFTGUARD_AUTO_RETRAIN", "true").lower() in ("1", "true", "yes"),
            webhook_url=g("DRIFTGUARD_WEBHOOK", ""),
            slack_webhook=g("SLACK_WEBHOOK", ""),
        )
