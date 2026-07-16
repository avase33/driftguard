"""Synthetic credit-card fraud transactions with a distribution-shifted variant.

`generate(n, drift=False)` produces the **reference** distribution the model is
trained on. `generate(n, drift=True)` simulates the world three months later: a
new fraud pattern has emerged, shifting several feature distributions (amount,
distance, merchant risk, foreign/online share) *and* the fraud-generating
relationship — so the old model degrades and the Population Stability Index picks
up the change. Deterministic given the seed.
"""

from __future__ import annotations

import math
import random
from typing import Any

from ..models import FEATURES, Dataset


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _row(rng: random.Random, drift: bool) -> tuple[list[float], int]:
    if not drift:
        amount = math.exp(rng.gauss(3.4, 0.9))
        hour = rng.choices(range(24), weights=[2 if 8 <= h <= 21 else 1 for h in range(24)])[0]
        distance = rng.expovariate(1 / 8.0)
        txn_last_hour = min(10, int(rng.expovariate(1 / 1.2)))
        merchant_risk = rng.betavariate(2, 6)          # low, mean ~0.25
        account_age = rng.uniform(60, 3000)
        foreign = 1 if rng.random() < 0.05 else 0
        online = 1 if rng.random() < 0.35 else 0
        night = 1 if (hour < 6 or hour > 22) else 0
        amount_z = math.log1p(amount) - 3.4
        # reference fraud: driven by merchant risk, foreign, night, big/far spend.
        # A sharp (bimodal) signal so clear-cut cases dominate and the model learns well.
        risk = (3.2 * (merchant_risk - 0.28)
                + 1.4 * foreign
                + 1.1 * night
                + 0.9 * max(0.0, amount_z)
                + 1.0 * min(distance, 60) / 60.0)
        logit = -3.0 + 2.6 * risk - 0.0004 * account_age
    else:
        # three months later: distributions shift AND the fraud mechanism changes
        # to a new online micro-transaction / velocity pattern.
        amount = math.exp(rng.gauss(4.1, 1.0))
        hour = rng.choices(range(24), weights=[2 if (h < 6 or h > 20) else 1 for h in range(24)])[0]
        distance = rng.expovariate(1 / 35.0)
        txn_last_hour = min(18, int(rng.expovariate(1 / 2.8)))
        merchant_risk = rng.betavariate(4, 4)          # higher, mean ~0.5
        account_age = rng.uniform(1, 700)
        foreign = 1 if rng.random() < 0.25 else 0
        online = 1 if rng.random() < 0.70 else 0
        night = 1 if (hour < 6 or hour > 22) else 0
        micro = 1 if (online and amount < 45) else 0
        # merchant_risk / foreign matter far less now; a reference-trained model misfires
        logit = (-3.0
                 + 1.0 * (merchant_risk - 0.5)
                 + 0.4 * foreign
                 + 3.2 * micro
                 + 1.6 * min(txn_last_hour, 15) / 15.0
                 + 0.8 * night
                 - 0.0002 * account_age)

    p = _sigmoid(logit)
    y = 1 if rng.random() < p else 0
    row = [round(amount, 2), float(hour), round(distance, 2), float(txn_last_hour),
           round(merchant_risk, 4), round(account_age, 1), float(foreign), float(online)]
    return row, y


def generate(n: int, seed: int = 7, drift: bool = False) -> Dataset:
    rng = random.Random(seed + (1000 if drift else 0))
    X, y = [], []
    for _ in range(n):
        row, label = _row(rng, drift)
        X.append(row)
        y.append(label)
    return Dataset(X=X, y=y, feature_names=list(FEATURES))


def transaction_dict(row: list[float]) -> dict[str, Any]:
    return dict(zip(FEATURES, row))
