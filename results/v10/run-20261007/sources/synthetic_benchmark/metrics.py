"""Общая ROC-AUC для сохранённых исследовательских прогнозов."""

import numpy as np


def compute_roc_auc(y_true: list[int], y_scores: list[float]) -> float:
    """Вычисляет ROC-AUC через U-критерий Манна-Уитни."""
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_scores))
    except Exception:
        pass

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pairs = sorted(zip(y_scores, y_true), key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n - 1 and pairs[j + 1][0] == pairs[j][0]:
            j += 1
        avg_rank = 1.0 + (i + j) / 2.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1

    u_stat = rank_sum - (n_pos * (n_pos + 1)) / 2.0
    return float(np.clip(u_stat / (n_pos * n_neg), 0.0, 1.0))


