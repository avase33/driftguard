"""Exception hierarchy for DriftGuard."""

from __future__ import annotations


class DriftGuardError(Exception):
    """Base class for all DriftGuard errors."""


class ConfigError(DriftGuardError):
    """Invalid configuration."""


class NotTrainedError(DriftGuardError):
    """A model was used before being trained or loaded."""


class RegistryError(DriftGuardError):
    """Model registry failure."""


class StreamError(DriftGuardError):
    """Streaming/queue transport failure."""
