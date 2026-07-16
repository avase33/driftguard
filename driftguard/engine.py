"""DriftGuard — the engine wiring the four operational zones together.

    Ingestion (producer -> bus) -> Serving (predict from latest model)
                                -> Monitoring (batch -> PSI drift check)
                                -> Retraining (on drift: retrain -> register -> reload)

One object is shared by the CLI, API and tests. Offline it needs no external
services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .alerting import build_dispatcher
from .config import Settings
from .drift.detector import DriftDetector
from .logging_setup import get_logger
from .model.gbdt import build_model
from .model.metrics import classification_metrics
from .models import Dataset, DriftReport, Prediction
from .monitoring import MonitoringObserver
from .registry import build_registry
from .retraining import RetrainOrchestrator
from .streaming.bus import build_bus
from .streaming.producer import TransactionProducer

log = get_logger("engine")


@dataclass
class CycleResult:
    report: DriftReport
    retrained: bool
    new_version: Optional[int] = None
    prev_f1: float = 0.0
    new_f1: float = 0.0


class DriftGuard:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        s = self.settings
        self.registry = build_registry(s.registry_backend, s.registry_dir, s.mlflow_uri)
        self.dispatcher = build_dispatcher(s.webhook_url, s.slack_webhook)
        self.orchestrator = RetrainOrchestrator(self.registry, s)
        self.bus = build_bus(s.bus_backend, s.kafka_brokers, s.topic, s.partitions)
        self.producer = TransactionProducer(self.bus)
        self.model = None
        self.reference: Optional[Dataset] = None
        self.detector: Optional[DriftDetector] = None
        self.monitor: Optional[MonitoringObserver] = None

    # ---- foundation: train baseline ------------------------------------

    def fit_reference(self, train: Dataset, test: Optional[Dataset] = None) -> dict:
        s = self.settings
        test = test or train
        model_params = {"n_estimators": s.n_estimators, "max_depth": s.max_depth,
                        "learning_rate": s.learning_rate, "reg_lambda": s.reg_lambda,
                        "min_child_weight": s.min_child_weight, "max_bins": s.max_bins}
        params = {"backend": s.model_backend, **model_params}
        model = build_model(s.model_backend, **model_params)
        model.fit(train.X, train.y)
        metrics = classification_metrics(test.y, model.predict_proba(test.X))
        self.registry.log_model(model, metrics, params, reason="initial")
        self.model = model
        self.reference = train
        self.detector = DriftDetector(train, s)
        self.monitor = MonitoringObserver(self.detector, self.dispatcher, s.monitor_batch)
        log.info("baseline model v%d trained: F1=%.3f AUC=%.3f",
                 self.registry.latest_version(), metrics["f1"], metrics["auc"])
        return metrics

    # ---- serving --------------------------------------------------------

    def predict(self, features) -> Prediction:
        if self.model is None:
            raise RuntimeError("no model — call fit_reference first")
        p = self.model.predict_proba_one(features)
        return Prediction(proba=p, label=1 if p >= 0.5 else 0,
                          model_version=self.registry.latest_version())

    def evaluate(self, ds: Dataset) -> dict:
        return classification_metrics(ds.y, self.model.predict_proba(ds.X))

    # ---- ingestion ------------------------------------------------------

    def stream(self, ds: Dataset, rate_per_sec=None, limit=None) -> int:
        return self.producer.stream(ds, rate_per_sec=rate_per_sec, limit=limit)

    def consume(self, max_records: int = 100000) -> list[dict]:
        return self.bus.poll(max_records)

    # ---- monitoring + retraining loop ----------------------------------

    def run_cycle(self, current: Dataset, test: Optional[Dataset] = None) -> CycleResult:
        """Run monitoring on a batch and auto-retrain if it has drifted."""
        assert self.detector is not None and self.monitor is not None
        report = self.detector.detect(current.X)
        self.monitor.history.append(report)
        self.monitor.stats.batches_checked += 1
        self.monitor.stats.last_drift_share = report.drift_share

        if not report.dataset_drift:
            return CycleResult(report=report, retrained=False)

        self.monitor.stats.drift_events += 1
        from .alerting import alert_from_report
        self.dispatcher(alert_from_report(report))

        if not self.settings.auto_retrain:
            return CycleResult(report=report, retrained=False)

        prev_f1 = self.evaluate(test or current)["f1"]
        result = self.orchestrator.retrain(current, test or current, reason="drift", prev_f1=prev_f1)
        # hot-reload latest model + rebase monitoring to the new normal
        _, self.model = self.registry.latest()
        self.reference = current
        self.detector = DriftDetector(current, self.settings)
        self.monitor.set_reference(self.detector)
        return CycleResult(report=report, retrained=True, new_version=result.version.version,
                           prev_f1=result.prev_f1, new_f1=result.new_f1)

    # ---- status ---------------------------------------------------------

    def status(self) -> dict:
        versions = self.registry.versions()
        return {
            "model_version": self.registry.latest_version(),
            "total_versions": len(versions),
            "reference_size": len(self.reference) if self.reference else 0,
            "bus_lag": self.bus.lag(),
            "monitor": {
                "batches_checked": self.monitor.stats.batches_checked if self.monitor else 0,
                "drift_events": self.monitor.stats.drift_events if self.monitor else 0,
                "last_drift_share": round(self.monitor.stats.last_drift_share, 3) if self.monitor else 0.0,
            },
            "versions": [v.to_dict() for v in versions],
        }
