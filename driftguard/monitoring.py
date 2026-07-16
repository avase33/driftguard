"""Monitoring zone.

An observer that batches the live stream and, once it has enough samples,
compares them against the training reference with the drift detector. On dataset
drift it dispatches an alert (which, in production, triggers the retraining
workflow). Keeps a rolling history of drift reports for the dashboard.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from .alerting import Alert, alert_from_report
from .drift.detector import DriftDetector
from .logging_setup import get_logger
from .models import DriftReport

log = get_logger("monitoring")


@dataclass
class MonitorStats:
    batches_checked: int = 0
    drift_events: int = 0
    last_drift_share: float = 0.0


class MonitoringObserver:
    def __init__(self, detector: DriftDetector, dispatcher: Callable[[Alert], str],
                 batch_size: int = 500) -> None:
        self.detector = detector
        self.dispatcher = dispatcher
        self.batch_size = batch_size
        self._buffer: list[list[float]] = []
        self.history: deque = deque(maxlen=100)
        self.stats = MonitorStats()

    def observe(self, rows) -> Optional[DriftReport]:
        """Accumulate rows; when a full batch arrives, run a drift check."""
        self._buffer.extend(rows)
        if len(self._buffer) < self.batch_size:
            return None
        return self._check()

    def flush(self) -> Optional[DriftReport]:
        if not self._buffer:
            return None
        return self._check()

    def _check(self) -> DriftReport:
        report = self.detector.detect(self._buffer)
        self._buffer = []
        self.stats.batches_checked += 1
        self.stats.last_drift_share = report.drift_share
        self.history.append(report)
        if report.dataset_drift:
            self.stats.drift_events += 1
            self.dispatcher(alert_from_report(report))
        return report

    def set_reference(self, detector: DriftDetector) -> None:
        """Rebase monitoring on a new normal after a retrain."""
        self.detector = detector
