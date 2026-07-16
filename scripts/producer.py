#!/usr/bin/env python3
"""Kafka producer — stream transactions row-by-row (Phase 3).

    python scripts/producer.py                 # stream reference then drifted data
    python scripts/producer.py --drift -n 500  # stream 500 drifted rows
    python scripts/producer.py --rate 1        # ~1 event/sec (live feed simulation)
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Stream synthetic transactions to the bus")
    ap.add_argument("-n", "--rows", type=int, default=1000)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--rate", type=float, default=None, help="events/sec (default: as fast as possible)")
    args = ap.parse_args(argv)

    from driftguard.config import Settings
    from driftguard.data.mockdata import generate
    from driftguard.streaming.bus import build_bus
    from driftguard.streaming.producer import TransactionProducer

    s = Settings()
    bus = build_bus(s.bus_backend, s.kafka_brokers, s.topic, s.partitions)
    producer = TransactionProducer(bus)
    ds = generate(args.rows, seed=10 if args.drift else 7, drift=args.drift)
    n = producer.stream(ds, rate_per_sec=args.rate)
    print(f"Produced {n} {'drifted' if args.drift else 'reference'} transactions "
          f"to topic '{s.topic}' (bus lag {bus.lag()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
