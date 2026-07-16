"""Streaming ingestion: Kafka-style bus + transaction producer."""

from .bus import PartitionedBus, build_bus
from .producer import TransactionProducer

__all__ = ["PartitionedBus", "build_bus", "TransactionProducer"]
