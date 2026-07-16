"""Classification metrics (from scratch)."""

from __future__ import annotations

from typing import Sequence


def classification_metrics(y_true: Sequence[int], proba: Sequence[float], threshold: float = 0.5) -> dict:
    tp = fp = tn = fn = 0
    for yt, p in zip(y_true, proba):
        pred = 1 if p >= threshold else 0
        if pred == 1 and yt == 1:
            tp += 1
        elif pred == 1 and yt == 0:
            fp += 1
        elif pred == 0 and yt == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "auc": roc_auc(y_true, proba)}


def roc_auc(y_true: Sequence[int], proba: Sequence[float]) -> float:
    """Rank-based (Mann-Whitney U) AUC."""
    paired = sorted(zip(proba, y_true), key=lambda p: p[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    i, n = 0, len(paired)
    while i < n:
        j = i
        while j < n and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if paired[k][1] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
