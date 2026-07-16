"""Transaction producer — the data producer of the ingestion zone.

Streams rows from a dataset into the bus one event at a time (optionally throttled
to ~1 event/sec to mimic a live feed), keyed so each account's transactions keep
ordering. Used to replay the reference stream and then the drifted stream.
"""

from __future__ import annotations

import time
from typing import Optional

from ..data.mockdata import transaction_dict
from ..models import Dataset
from .bus import PartitionedBus


class TransactionProducer:
    def __init__(self, bus: PartitionedBus) -> None:
        self.bus = bus

    def stream(self, dataset: Dataset, rate_per_sec: Optional[float] = None,
               limit: Optional[int] = None) -> int:
        delay = (1.0 / rate_per_sec) if rate_per_sec else 0.0
        n = 0
        for i, (row, label) in enumerate(zip(dataset.X, dataset.y)):
            if limit is not None and n >= limit:
                break
            key = f"acct_{i % 500}"
            msg = {"features": row, "label": label, "txn": transaction_dict(row)}
            self.bus.produce(key, msg)
            n += 1
            if delay:
                time.sleep(delay)
        return n
