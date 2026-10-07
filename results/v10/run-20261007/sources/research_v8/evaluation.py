"""Macro-метрики и парные пользовательские интервалы исследовательского V8.1."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import beta

from research_v7.evaluation import metrics, notification_metrics


QUALITY_METRICS = ("recall", "false_positive_rate", "youden_j", "brier", "auc", "candidate_brier", "candidate_auc")
METRICS_PROTOCOL_URL = "https://github.com/nutrifit-health/craving-engine/blob/main/docs/protocol-v9.md#metrics"


def with_youden_j(result: dict) -> dict:
    """Общий коэффициент определён только при наличии обоих классов исхода.

    J=TPR−FPR; идеал1, случайное независимое предупреждение в среднем0.
    Это не accuracy. Один процентный пункт TPR и FPR имеет равный вес.
    """
    recall, fpr = result["recall"], result["false_positive_rate"]
    value = None if recall is None or fpr is None else float(recall - fpr)
    return {**result, "youden_j": value}


def _by_user(rows: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        result[row["user_id"]].append(row)
    return dict(result)


def interval(values: list[float], seed: int = 20260923, draws: int = 2000) -> dict:
    """Повторная выборка людей; дни не считаются независимыми наблюдениями."""
    if not values:
        return {"mean": None, "ci95": None, "upper95": None, "users": 0}
    value = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {"mean": float(value.mean()), "ci95": None, "upper95": None, "users": len(values)}
    indices = np.random.default_rng(seed).integers(len(value), size=(draws, len(value)))
    means = value[indices].mean(axis=1)
    return {"mean": float(value.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(), "upper95": float(np.quantile(means, .95)), "users": len(values)}


def choose_thresholds(calibration_rows: list[dict]) -> dict[str, float]:
    """Максимизирует macro recall только на calibration при macro FPR≤20%."""
    from fractions import Fraction

    if not calibration_rows or any(row["split"] != "calibration" for row in calibration_rows):
        raise ValueError("Выбор порога разрешён только на calibration")
    groups = _by_user(calibration_rows)
    if {row["outcome"] for row in calibration_rows if row["outcome"] is not None} != {0, 1}:
        raise ValueError("Calibration требует оба исхода")
    counts = {
        user: (sum(row["outcome"] == 1 for row in rows), sum(row["outcome"] == 0 for row in rows))
        for user, rows in groups.items()
    }
    positive_users = sum(positive > 0 for positive, _ in counts.values())
    negative_users = sum(negative > 0 for _, negative in counts.values())
    # Рациональные веса сохраняют ровно 20%/94% и одинаковый tie-break независимо
    # от накопления floating-point ошибки. Каждый пользователь имеет равный вес.
    increments = []
    for user, rows in groups.items():
        positive, negative = counts[user]
        positive_weight = Fraction(1, positive_users * positive) if positive else Fraction(0)
        negative_weight = Fraction(1, negative_users * negative) if negative else Fraction(0)
        for row in rows:
            probability = float(row["probability"])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Calibration содержит некорректную вероятность")
            increments.append((
                probability,
                positive_weight if row["outcome"] == 1 else Fraction(0),
                negative_weight if row["outcome"] == 0 else Fraction(0),
            ))
    increments.sort(key=lambda item: item[0], reverse=True)
    # Конечный порог совместим с JSON и исключает все вероятности в [0,1].
    no_alert = 1.0 if increments[0][0] < 1.0 else float(np.nextafter(1.0, np.inf))
    recall, fpr = Fraction(0), Fraction(0)
    grid = [(no_alert, recall, fpr)]
    index = 0
    while index < len(increments):
        threshold = increments[index][0]
        # Все равные scores включаются вместе, в соответствии с probability >= threshold.
        while index < len(increments) and increments[index][0] == threshold:
            recall += increments[index][1]
            fpr += increments[index][2]
            index += 1
        grid.append((threshold, recall, fpr))
    acceptable = [r for r in grid if r[2] <= Fraction(1, 5)]
    if not acceptable:
        raise ValueError("Не найден порог с допустимой частотой ложных тревог")
    moderate = max(acceptable, key=lambda r: (r[1], -r[2], r[0]))
    sensitive = max((r for r in grid if r[1] >= Fraction(94, 100)), key=lambda r: (-r[2], r[0]))
    return {"fixed_030": .3, "calibration_macro_fpr20": moderate[0], "calibration_macro_recall94": sensitive[0]}


def model_report(rows: list[dict], thresholds: dict[str, float], training_prevalence: float) -> dict:
    """В отчёте одинаково видны все пользователи, пропуски и обе группы ошибок."""
    groups = _by_user(rows)
    operating_points = {}
    for name, threshold in thresholds.items():
        people = {user: with_youden_j(metrics(items, threshold, training_prevalence)) for user, items in sorted(groups.items())}
        # J усредняется по людям с обоими классами, а не вычитается из двух
        # средних recall/FPR, состав пользователей которых может различаться.
        macro = {key: interval([p[key] for p in people.values() if p[key] is not None]) for key in QUALITY_METRICS}
        operating_points[name] = {"threshold": threshold, "per_user": people, "macro": macro, "pooled": with_youden_j(metrics(rows, threshold, training_prevalence)), "notifications": {str(limit): notification_metrics(rows, threshold, limit) for limit in (2, 3, 7)}}
    return {"operating_points": operating_points, "users": len(groups), "decisions": len(rows), "metrics_protocol": METRICS_PROTOCOL_URL, "metric_definitions": {"youden_j": {"formula": "recall - false_positive_rate", "range": [-1, 1], "higher_is_better": True, "unit": "dimensionless_not_accuracy", "macro_population": "users with both outcome classes", "policy_scope": "classifier threshold before notification limits"}}}


def paired_comparison(candidate: dict, baseline: dict, point: str = "calibration_macro_fpr20") -> dict:
    """Дельты считаются на одних пользователях и одном наборе исходов."""
    c = {user: with_youden_j(value) for user, value in candidate["operating_points"][point]["per_user"].items()}
    b = {user: with_youden_j(value) for user, value in baseline["operating_points"][point]["per_user"].items()}
    if set(c) != set(b):
        raise ValueError("Парное сравнение требует одинаковых пользователей")
    deltas = {key: [] for key in QUALITY_METRICS}
    for user in sorted(c):
        if c[user]["events"] != b[user]["events"] or c[user]["scored"] != b[user]["scored"]:
            raise ValueError("В сравниваемых моделях расходятся исходы")
        for key in deltas:
            if c[user][key] is not None and b[user][key] is not None:
                deltas[key].append(c[user][key] - b[user][key])
    result = {key + "_delta": interval(values) for key, values in deltas.items()}
    count = len(deltas["auc"])
    degraded = sum(delta < -.01 for delta in deltas["auc"])
    result["individual_auc_degradation"] = {
        "users": count, "count": degraded, "fraction": degraded / count if count else None,
        "upper95_exact_binomial": (1.0 if degraded == count else float(beta.ppf(.95, degraded + 1, count - degraded))) if count else None,
        "definition": "Наблюдаемая индивидуальная AUC-разность < −0.01; не оценка истинного вреда",
    }
    result.update({"bootstrap_unit": "user", "bootstrap_draws": 2000, "seed": 20260923, "operating_point": point, "status": "exploratory_known_development_not_confirmatory"})
    return result
