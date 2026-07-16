"""Drift detector.

Holds the training **reference** data and, given a batch of live production rows,
computes per-feature PSI and decides whether the *dataset* has drifted: a feature
is "drifted" when its PSI ≥ the alert threshold, and the pipeline fires when more
than ``drift_feature_fraction`` of features have drifted. This is the signal that
triggers retraining. An Evidently adapter can produce the same verdict with rich
HTML reports.
"""

from __future__ import annotations

from typing import Sequence

from ..config import Settings
from ..models import Dataset, DriftReport, FeatureDrift
from .psi import population_stability_index


class DriftDetector:
    def __init__(self, reference: Dataset, settings: Settings | None = None) -> None:
        self.reference = reference
        self.settings = settings or Settings()
        self._ref_cols = {name: reference.column(name) for name in reference.feature_names}

    def detect(self, rows: Sequence[Sequence[float]]) -> DriftReport:
        s = self.settings
        names = self.reference.feature_names
        features: list[FeatureDrift] = []
        drifted = 0
        for i, name in enumerate(names):
            cur = [row[i] for row in rows]
            psi = population_stability_index(self._ref_cols[name], cur, bins=s.psi_bins)
            is_drift = psi >= s.psi_alert
            drifted += int(is_drift)
            features.append(FeatureDrift(feature=name, psi=psi, drifted=is_drift))

        share = drifted / len(names) if names else 0.0
        dataset_drift = share > s.drift_feature_fraction
        return DriftReport(features=features, drift_share=share,
                           dataset_drift=dataset_drift, n_samples=len(rows))

    def detect_dataset(self, ds: Dataset) -> DriftReport:
        return self.detect(ds.X)
