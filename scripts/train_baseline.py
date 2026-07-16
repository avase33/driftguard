#!/usr/bin/env python3
"""Train the baseline fraud model and register it (Phase 1).

    python scripts/train_baseline.py            # train + register v1, print metrics
    python scripts/train_baseline.py --persist  # also write versions to the registry dir
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Train the DriftGuard baseline model")
    ap.add_argument("-n", "--rows", type=int, default=5000)
    ap.add_argument("--persist", action="store_true")
    args = ap.parse_args(argv)

    from driftguard.config import Settings
    from driftguard.data.mockdata import generate
    from driftguard.engine import DriftGuard

    settings = Settings()
    settings.registry_backend = "local"
    dg = DriftGuard(settings)
    if args.persist:
        dg.registry.persist = True
        import os
        os.makedirs(dg.registry.directory, exist_ok=True)

    metrics = dg.fit_reference(generate(args.rows, seed=7), generate(max(500, args.rows // 3), seed=8))
    print("Baseline model registered as v%d" % dg.registry.latest_version())
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
