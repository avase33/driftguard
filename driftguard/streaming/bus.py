"""Partitioned message bus (Kafka-style) — the ingestion zone's transport.

The producer streams transactions into partitions chosen by a stable key hash;
the serving zone consumes them for inference and the monitoring zone batches them
for drift checks. The default :class:`PartitionedBus` is an in-memory model with
identical semantics; :class:`KafkaBus` is the production adapter.
"""

from __future__ import annotations

import hashlib
import threading
from collections import deque


def partition_for(key: str, partitions: int) -> int:
    h = int.from_bytes(hashlib.md5(key.encode()).digest()[:4], "big")
    return h % partitions


class PartitionedBus:
    def __init__(self, partitions: int = 4) -> None:
        self.partitions = partitions
        self._queues: list[deque] = [deque() for _ in range(partitions)]
        self._produced = 0
        self._lock = threading.Lock()

    def produce(self, key: str, message: dict) -> int:
        p = partition_for(key, self.partitions)
        with self._lock:
            self._queues[p].append(message)
            self._produced += 1
        return p

    def poll(self, max_records: int = 1000) -> list[dict]:
        out: list[dict] = []
        for q in self._queues:
            while q and len(out) < max_records:
                out.append(q.popleft())
        return out

    def lag(self) -> int:
        return sum(len(q) for q in self._queues)

    def total_produced(self) -> int:
        return self._produced


class KafkaBus:  # pragma: no cover - requires kafka
    def __init__(self, brokers: str, topic: str = "transactions", partitions: int = 4) -> None:
        from kafka import KafkaProducer  # type: ignore
        import json

        self.topic = topic
        self.partitions = partitions
        self._producer = KafkaProducer(bootstrap_servers=brokers,
                                       value_serializer=lambda v: json.dumps(v).encode(),
                                       key_serializer=lambda k: k.encode())

    def produce(self, key: str, message: dict) -> int:
        fut = self._producer.send(self.topic, key=key, value=message)
        return fut.get(timeout=10).partition

    def poll(self, max_records: int = 1000):
        raise NotImplementedError("KafkaBus is consumed by dedicated consumers")


def build_bus(backend: str = "memory", brokers: str = "", topic: str = "transactions",
              partitions: int = 4) -> PartitionedBus:
    if backend == "kafka" and brokers:
        return KafkaBus(brokers, topic, partitions)  # type: ignore[return-value]
    return PartitionedBus(partitions)
