"""Gradient-Boosted Decision Trees — XGBoost-style, from scratch.

A histogram-based gradient booster for binary classification (log-loss). It is a
faithful, compact implementation of the core XGBoost algorithm:

* features are quantile-**binned** once, so split finding scans histograms, not raw
  values (the key to XGBoost's speed);
* each round fits a regression tree to the gradient/hessian of the log-loss with
  the second-order split gain
  ``0.5 * (GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ))`` and leaf values ``-G/(H+λ)``;
* ``min_child_weight`` (min hessian per leaf) and ``reg_lambda`` regularise it.

Pure Python, no numpy — trains a solid fraud classifier on a few thousand rows in
well under a second. A drop-in :class:`XGBoostModel` adapter uses real XGBoost.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..errors import NotTrainedError


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class GradientBoostedTrees:
    n_estimators: int = 60
    max_depth: int = 3
    learning_rate: float = 0.3
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    max_bins: int = 32
    min_split_gain: float = 0.0

    base_score: float = 0.0
    trees: list = field(default_factory=list)
    _edges: list = field(default_factory=list)          # per-feature bin edges
    _n_features: int = 0
    _trained: bool = False

    # ---- binning --------------------------------------------------------

    def _fit_bins(self, X: Sequence[Sequence[float]]) -> None:
        self._n_features = len(X[0])
        self._edges = []
        n = len(X)
        for f in range(self._n_features):
            vals = sorted(row[f] for row in X)
            edges: list[float] = []
            for b in range(1, self.max_bins):
                q = vals[min(n - 1, int(b / self.max_bins * n))]
                if not edges or q > edges[-1]:
                    edges.append(q)
            self._edges.append(edges)

    def _bin_row(self, row: Sequence[float]) -> list[int]:
        return [bisect.bisect_right(self._edges[f], row[f]) for f in range(self._n_features)]

    # ---- training -------------------------------------------------------

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "GradientBoostedTrees":
        if not X:
            raise NotTrainedError("empty training set")
        self._fit_bins(X)
        binX = [self._bin_row(r) for r in X]
        y = list(y)
        n = len(y)
        pos_rate = max(1e-6, min(1 - 1e-6, sum(y) / n))
        self.base_score = math.log(pos_rate / (1 - pos_rate))
        F = [self.base_score] * n
        self.trees = []

        for _ in range(self.n_estimators):
            g = [0.0] * n
            h = [0.0] * n
            for i in range(n):
                p = _sigmoid(F[i])
                g[i] = p - y[i]
                h[i] = max(1e-6, p * (1 - p))
            tree = self._build_tree(list(range(n)), binX, g, h, depth=0)
            for i in range(n):
                F[i] += self.learning_rate * _leaf_value(tree, binX[i])
            self.trees.append(tree)

        self._trained = True
        return self

    def _build_tree(self, rows, binX, g, h, depth):
        G = sum(g[i] for i in rows)
        H = sum(h[i] for i in rows)
        if depth >= self.max_depth or len(rows) < 2:
            return {"leaf": -G / (H + self.reg_lambda)}

        best = None  # (gain, feature, bin_threshold)
        base = G * G / (H + self.reg_lambda)
        for f in range(self._n_features):
            nbins = len(self._edges[f]) + 1
            hg = [0.0] * nbins
            hh = [0.0] * nbins
            for i in rows:
                b = binX[i][f]
                hg[b] += g[i]
                hh[b] += h[i]
            GL = HL = 0.0
            for t in range(nbins - 1):
                GL += hg[t]; HL += hh[t]
                GR = G - GL; HR = H - HL
                if HL < self.min_child_weight or HR < self.min_child_weight:
                    continue
                gain = 0.5 * (GL * GL / (HL + self.reg_lambda)
                              + GR * GR / (HR + self.reg_lambda) - base)
                if best is None or gain > best[0]:
                    best = (gain, f, t)

        if best is None or best[0] <= self.min_split_gain:
            return {"leaf": -G / (H + self.reg_lambda)}

        _, f, t = best
        left = [i for i in rows if binX[i][f] <= t]
        right = [i for i in rows if binX[i][f] > t]
        if not left or not right:
            return {"leaf": -G / (H + self.reg_lambda)}
        return {"f": f, "t": t,
                "l": self._build_tree(left, binX, g, h, depth + 1),
                "r": self._build_tree(right, binX, g, h, depth + 1)}

    # ---- inference ------------------------------------------------------

    def _raw(self, row: Sequence[float]) -> float:
        if not self._trained:
            raise NotTrainedError("model not trained")
        b = self._bin_row(row)
        score = self.base_score
        for tree in self.trees:
            score += self.learning_rate * _leaf_value(tree, b)
        return score

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        return [_sigmoid(self._raw(r)) for r in X]

    def predict_proba_one(self, row: Sequence[float]) -> float:
        return _sigmoid(self._raw(row))

    def predict(self, X: Sequence[Sequence[float]], threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    # ---- (de)serialisation for the registry -----------------------------

    def to_state(self) -> dict:
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "learning_rate": self.learning_rate, "reg_lambda": self.reg_lambda,
                "min_child_weight": self.min_child_weight, "max_bins": self.max_bins,
                "base_score": self.base_score, "edges": self._edges,
                "n_features": self._n_features, "trees": self.trees}

    @classmethod
    def from_state(cls, s: dict) -> "GradientBoostedTrees":
        m = cls(n_estimators=s["n_estimators"], max_depth=s["max_depth"],
                learning_rate=s["learning_rate"], reg_lambda=s["reg_lambda"],
                min_child_weight=s["min_child_weight"], max_bins=s["max_bins"])
        m.base_score = s["base_score"]
        m._edges = s["edges"]
        m._n_features = s["n_features"]
        m.trees = s["trees"]
        m._trained = True
        return m


def _leaf_value(node, binrow) -> float:
    while "leaf" not in node:
        node = node["l"] if binrow[node["f"]] <= node["t"] else node["r"]
    return node["leaf"]


class XGBoostModel:  # pragma: no cover - optional dep
    def __init__(self, **params) -> None:
        import xgboost as xgb  # type: ignore

        self._xgb = xgb
        self._params = {"max_depth": params.get("max_depth", 3),
                        "eta": params.get("learning_rate", 0.3),
                        "objective": "binary:logistic", "eval_metric": "logloss"}
        self._n = params.get("n_estimators", 60)
        self._model = None

    def fit(self, X, y):
        d = self._xgb.DMatrix(list(X), label=list(y))
        self._model = self._xgb.train(self._params, d, num_boost_round=self._n)
        return self

    def predict_proba(self, X):
        d = self._xgb.DMatrix(list(X))
        return [float(p) for p in self._model.predict(d)]

    def predict_proba_one(self, row):
        return self.predict_proba([row])[0]

    def predict(self, X, threshold: float = 0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]


def build_model(backend: str = "gbdt", **params):
    if backend == "xgboost":
        return XGBoostModel(**params)
    return GradientBoostedTrees(
        n_estimators=params.get("n_estimators", 60), max_depth=params.get("max_depth", 3),
        learning_rate=params.get("learning_rate", 0.3), reg_lambda=params.get("reg_lambda", 1.0),
        min_child_weight=params.get("min_child_weight", 1.0), max_bins=params.get("max_bins", 32))
