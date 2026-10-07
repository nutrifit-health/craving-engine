"""Парные метрики и кривые адаптации; пользователь является единицей анализа."""

from collections import defaultdict

import numpy as np

from research_v8.evaluation import interval, model_report, paired_comparison

POINT = "calibration_macro_fpr20"


def macro_brier(rows: list[dict]) -> float:
    errors = defaultdict(list)
    for row in rows:
        errors[row["user_id"]].append((row["probability"] - row["outcome"]) ** 2)
    return float(np.mean([np.mean(values) for values in errors.values()]))


def report(rows: list[dict], threshold: float, prevalence: float = 0.5) -> dict:
    return model_report(rows, {POINT: threshold}, prevalence)


def comparison(candidate: dict, baseline: dict) -> dict:
    return paired_comparison(candidate, baseline, POINT)


def adaptation(rows: list[dict]) -> dict:
    """Точные числа уже полученных post-change ответов, не номера дней."""
    result = {}
    for count in (0, 1, 2, 5, 10, 20):
        people = defaultdict(list)
        for row in rows:
            if (
                row["change_day"] is not None
                and row["day"] >= row["change_day"]
                and row["post_change_feedback"] == count
            ):
                people[row["user_id"]].append(row)
        result[str(count)] = {
            "users": len(people),
            "decisions": sum(map(len, people.values())),
        }
        for name, field, truth in (
            ("brier_gated", "probability", "outcome"),
            ("brier_ungated", "ungated", "outcome"),
            ("probability_mse_gated", "probability", "true_probability"),
            ("probability_mse_ungated", "ungated", "true_probability"),
        ):
            values = [
                float(np.mean([(row[field] - row[truth]) ** 2 for row in items]))
                for items in people.values()
            ]
            result[str(count)][name] = interval(values, seed=20261007)
        result[str(count)]["gate_open_fraction"] = (
            float(np.mean([r["gamma"] > 0 for items in people.values() for r in items]))
            if people
            else None
        )
    return result


def compact_report(full: dict) -> dict:
    point = full["operating_points"][POINT]
    return {
        "users": full["users"],
        "decisions": full["decisions"],
        "threshold": point["threshold"],
        "macro": {k: v["mean"] for k, v in point["macro"].items()},
        "pooled": point["pooled"],
        "notifications": point["notifications"]["2"],
    }
