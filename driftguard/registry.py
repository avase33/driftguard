"""Model registry (MLflow-style).

Tracks every trained model as a numbered **version** with its metrics, params and
serialised weights — so serving always loads "the latest", retraining appends a
new version, and you can audit what changed. The default :class:`LocalRegistry`
persists to a JSON run directory (like a tiny ``mlruns/``); :class:`MLflowRegistry`
logs to a real MLflow tracking server behind the same interface.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from .errors import RegistryError
from .model.gbdt import GradientBoostedTrees
from .models import ModelVersion


class LocalRegistry:
    def __init__(self, directory: str = "mlruns_local", persist: bool = False) -> None:
        self.directory = directory
        self.persist = persist
        self._versions: list[ModelVersion] = []
        self._models: dict[int, GradientBoostedTrees] = {}
        if persist:
            os.makedirs(directory, exist_ok=True)
            self._load()

    def log_model(self, model: GradientBoostedTrees, metrics: dict, params: dict,
                  reason: str = "initial") -> ModelVersion:
        version = len(self._versions) + 1
        mv = ModelVersion(version=version, metrics=dict(metrics), params=dict(params), reason=reason)
        self._versions.append(mv)
        self._models[version] = model
        if self.persist:
            self._save(mv, model)
        return mv

    def latest(self) -> tuple[ModelVersion, GradientBoostedTrees]:
        if not self._versions:
            raise RegistryError("no model registered")
        mv = self._versions[-1]
        return mv, self._models[mv.version]

    def latest_version(self) -> int:
        return self._versions[-1].version if self._versions else 0

    def get(self, version: int) -> GradientBoostedTrees:
        if version not in self._models:
            raise RegistryError(f"unknown version {version}")
        return self._models[version]

    def versions(self) -> list[ModelVersion]:
        return list(self._versions)

    # ---- persistence ----------------------------------------------------

    def _path(self, version: int) -> str:
        return os.path.join(self.directory, f"v{version}.json")

    def _save(self, mv: ModelVersion, model: GradientBoostedTrees) -> None:
        with open(self._path(mv.version), "w", encoding="utf-8") as f:
            json.dump({"meta": mv.to_dict(), "model": model.to_state()}, f)

    def _load(self) -> None:
        files = sorted(fn for fn in os.listdir(self.directory)
                       if fn.startswith("v") and fn.endswith(".json"))
        for fn in files:
            with open(os.path.join(self.directory, fn), encoding="utf-8") as f:
                blob = json.load(f)
            meta = blob["meta"]
            mv = ModelVersion(version=meta["version"], metrics=meta["metrics"],
                              params=meta["params"], reason=meta.get("reason", ""),
                              created_at=meta.get("created_at", time.time()))
            self._versions.append(mv)
            self._models[mv.version] = GradientBoostedTrees.from_state(blob["model"])


class MLflowRegistry:  # pragma: no cover - requires mlflow
    def __init__(self, uri: str, experiment: str = "driftguard") -> None:
        import mlflow  # type: ignore

        self._mlflow = mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        self._local = LocalRegistry()   # weights kept locally; MLflow logs metadata

    def log_model(self, model, metrics, params, reason="initial"):
        with self._mlflow.start_run():
            self._mlflow.log_params(params)
            self._mlflow.log_metrics(metrics)
            self._mlflow.set_tag("reason", reason)
        return self._local.log_model(model, metrics, params, reason)

    def latest(self):
        return self._local.latest()

    def latest_version(self):
        return self._local.latest_version()

    def versions(self):
        return self._local.versions()


def build_registry(backend: str = "local", directory: str = "mlruns_local",
                   uri: str = "", persist: bool = False):
    if backend == "mlflow" and uri:
        return MLflowRegistry(uri)
    return LocalRegistry(directory, persist=persist)
