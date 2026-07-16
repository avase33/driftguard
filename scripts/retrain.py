#!/usr/bin/env python3
"""Retraining entrypoint — invoked by the GitHub Actions retrain workflow (Phase 5).

On a drift event this trains a fresh model on the newly-collected data, evaluates
it, and registers a new version. Prints a JSON summary the workflow can surface.
"""

from __future__ import annotations

import json
import sys


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    from driftguard.config import Settings
    from driftguard.data.mockdata import generate
    from driftguard.engine import DriftGuard

    dg = DriftGuard(Settings())
    dg.fit_reference(generate(4000, seed=7), generate(1000, seed=8))
    prev = dg.evaluate(generate(1000, seed=9, drift=True))["f1"]

    # in production this dataset comes from the collected drifted stream
    result = dg.run_cycle(generate(4000, seed=10, drift=True), generate(1000, seed=9, drift=True))
    print(json.dumps({
        "drift_detected": result.report.dataset_drift,
        "drift_share": round(result.report.drift_share, 3),
        "retrained": result.retrained,
        "new_version": result.new_version,
        "f1_before": round(prev, 4),
        "f1_after": round(result.new_f1, 4),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
