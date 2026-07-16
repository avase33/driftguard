"""Command-line interface for DriftGuard."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .config import Settings
from .data.mockdata import generate
from .engine import DriftGuard
from .logging_setup import configure_logging
from .models import FEATURES
from .version import __version__


def _reconfigure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def cmd_demo(args) -> int:
    n = args.rows
    dg = DriftGuard(Settings())

    train = generate(n, seed=7, drift=False)
    ref_test = generate(max(400, n // 3), seed=8, drift=False)
    drift_test = generate(max(400, n // 3), seed=9, drift=True)
    drift_train = generate(n, seed=10, drift=True)

    base = dg.fit_reference(train, ref_test)
    print("DriftGuard demo — automated drift detection & self-retraining\n")
    print(f"[1] Baseline model v1 trained on reference data")
    print(f"    reference F1={base['f1']:.3f}  AUC={base['auc']:.3f}  "
          f"precision={base['precision']:.3f}  recall={base['recall']:.3f}")

    degraded = dg.evaluate(drift_test)
    print(f"\n[2] Three months later — new fraud pattern emerges")
    print(f"    v1 F1 on drifted data dropped to {degraded['f1']:.3f} (AUC {degraded['auc']:.3f})")

    result = dg.run_cycle(drift_train, drift_test)
    r = result.report
    print(f"\n[3] Monitoring zone — PSI drift check ({r.n_samples} samples)")
    print(f"    dataset drift: {r.dataset_drift}  ({r.drift_share*100:.0f}% of features shifted)")
    for f in sorted(r.features, key=lambda x: x.psi, reverse=True):
        flag = "DRIFT" if f.drifted else "ok"
        print(f"      {f.feature:<20} PSI={f.psi:6.3f}  [{flag}]")

    if result.retrained:
        print(f"\n[4] Retraining zone — drift alert fired -> auto-retrained")
        print(f"    registered model v{result.new_version}; "
              f"F1 on drifted data recovered {result.prev_f1:.3f} -> {result.new_f1:.3f}")
    print(f"\nRegistry now has {dg.registry.latest_version()} model versions.")
    return 0


def cmd_predict(args) -> int:
    dg = DriftGuard(Settings())
    dg.fit_reference(generate(args.rows, seed=7))
    row = generate(1, seed=args.seed, drift=args.drift).X[0]
    pred = dg.predict(row)
    print(json.dumps({"transaction": dict(zip(FEATURES, row)), **pred.to_dict()}, indent=2))
    return 0


def cmd_status(args) -> int:
    dg = DriftGuard(Settings())
    dg.fit_reference(generate(args.rows, seed=7), generate(400, seed=8))
    dg.run_cycle(generate(args.rows, seed=10, drift=True), generate(400, seed=9, drift=True))
    print(json.dumps(dg.status(), indent=2))
    return 0


def cmd_serve(args) -> int:
    from .serving.api import run_server

    run_server(host=args.host, port=args.port, rows=args.rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="driftguard", description="Real-time drift detection & self-retraining pipeline")
    p.add_argument("--version", action="version", version=f"driftguard {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--rows", type=int, default=3000)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the full drift → alert → retrain lifecycle")
    d.set_defaults(func=cmd_demo)

    pr = sub.add_parser("predict", help="score a single (optionally drifted) transaction")
    pr.add_argument("--seed", type=int, default=1)
    pr.add_argument("--drift", action="store_true")
    pr.set_defaults(func=cmd_predict)

    s = sub.add_parser("status", help="print pipeline status + model versions JSON")
    s.set_defaults(func=cmd_status)

    sv = sub.add_parser("serve", help="run the FastAPI serving + monitoring dashboard")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    _reconfigure_stdout()
    args = build_parser().parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "WARNING")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
