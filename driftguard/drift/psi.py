"""Population Stability Index — computed from scratch.

PSI measures how much a feature's distribution has shifted between a reference
sample (training data) and a current sample (production data):

    PSI = Σ_i (Actual_i - Expected_i) · ln(Actual_i / Expected_i)

where Expected_i / Actual_i are the *fractions* of records falling in bin i for
the reference / current samples. Bins are defined once from the reference
distribution's quantiles so both samples are compared on the same edges.

Interpretation (industry standard):
    PSI < 0.10      no significant shift
    0.10 – 0.20     moderate shift — monitor
    PSI ≥ 0.20      significant drift — retrain
"""

from __future__ import annotations

import math
from typing import Sequence

_EPS = 1e-4


def psi_bins_from_reference(reference: Sequence[float], bins: int = 10) -> list[float]:
    """Quantile bin edges from the reference distribution."""
    vals = sorted(reference)
    n = len(vals)
    if n == 0:
        return []
    edges: list[float] = []
    for b in range(1, bins):
        q = vals[min(n - 1, int(b / bins * n))]
        if not edges or q > edges[-1]:
            edges.append(q)
    return edges


def _fractions(sample: Sequence[float], edges: list[float]) -> list[float]:
    import bisect
    counts = [0] * (len(edges) + 1)
    for v in sample:
        counts[bisect.bisect_right(edges, v)] += 1
    n = len(sample) or 1
    return [c / n for c in counts]


def population_stability_index(reference: Sequence[float], current: Sequence[float],
                               bins: int = 10) -> float:
    if not reference or not current:
        return 0.0
    edges = psi_bins_from_reference(reference, bins)
    expected = _fractions(reference, edges)
    actual = _fractions(current, edges)
    psi = 0.0
    for e, a in zip(expected, actual):
        e = max(e, _EPS)
        a = max(a, _EPS)
        psi += (a - e) * math.log(a / e)
    return psi
