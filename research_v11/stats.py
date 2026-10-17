"""Статистический анализ V11: парный bootstrap, односторонний тест под H0 и поправка Холма."""

from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Вычисляет средний Brier score (mean squared error вероятностей)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_pred, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def user_brier_scores(
    records: list[dict],
    predictions: np.ndarray,
) -> dict[str, float]:
    """Вычисляет средний Brier score для каждого пользователя."""
    by_user_y: dict[str, list[int]] = {}
    by_user_p: dict[str, list[float]] = {}

    for i, r in enumerate(records):
        u = r["user_id"]
        # Используем evaluation_outcome (истинный исход дня для оценки)
        y = r["evaluation_outcome"]
        p = float(predictions[i])
        by_user_y.setdefault(u, []).append(y)
        by_user_p.setdefault(u, []).append(p)

    return {
        u: brier_score(np.array(by_user_y[u]), np.array(by_user_p[u]))
        for u in sorted(by_user_y)
    }


def paired_bootstrap_test(
    diffs_per_user: np.ndarray,
    n_resamples: int = 20_000,
    seed: int = 20261105,
) -> dict[str, float | bool]:
    """Проводит парный bootstrap-анализ разностей среднего Brier по пользователям.

    Параметры:
        diffs_per_user: одномерный массив разностей Brier(A) - Brier(B) для каждого пользователя.
                        Отрицательное значение означает преимущество A над B.
        n_resamples: число реплик bootstrap (по протоколу 20 000).
        seed: фиксированный генератор случайных чисел.

    Возвращает:
        mean_diff: точечная оценка средней разности по пользователям;
        ci_lower, ci_upper: 95% процентильный доверительный интервал;
        p_value: односторонний p-value проверки H0: mu_diff >= 0 против H1: mu_diff < 0;
        n_users: число пользователей.
    """
    diffs = np.asarray(diffs_per_user, dtype=np.float64)
    n = len(diffs)
    if n < 3:
        raise ValueError("Требуется минимум 3 пользователя для bootstrap")

    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    # Генерация реплик для 95% ДИ
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(diffs[indices], axis=1)

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Односторонний тест под H0: центрирование разностей
    centered_diffs = diffs - mean_diff
    boot_h0_means = np.mean(centered_diffs[indices], axis=1)

    # p-value с поправкой Дейвисона-Хинкли (1 + sum(T* <= T_obs)) / (B + 1)
    # где меньшее значение свидетельствует в пользу H1 (mu < 0)
    hits = int(np.sum(boot_h0_means <= mean_diff))
    p_value = float((hits + 1) / (n_resamples + 1))

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_users": n,
        "ci_below_zero": bool(ci_upper < 0.0),
    }


def holm_bonferroni_correction(
    p_values: list[tuple[str, float]],
    alpha: float = 0.05,
) -> dict[str, dict[str, float | bool]]:
    """Применяет процедуру Холма-Бонферрони (step-down) для списка гипотез и p-values."""
    # Сортировка по возрастанию p-value
    m = len(p_values)
    sorted_p = sorted(p_values, key=lambda x: x[1])

    results = {}
    stopped = False
    for rank, (name, p) in enumerate(sorted_p, start=1):
        threshold = alpha / (m - rank + 1)
        if stopped or p > threshold:
            rejected = False
            stopped = True
        else:
            rejected = True
        results[name] = {
            "p_value": p,
            "rank": rank,
            "threshold": threshold,
            "significant": rejected,
        }
    return results
