"""Метрики предупреждений после применения лимита, отдельно от прогнозной модели."""

from collections import defaultdict
from datetime import datetime, timedelta
from itertools import pairwise

import numpy as np

from research_v13.policy import AlertPolicy
from research_v13.types import PolicyConfig


def _rates(counts: dict, user_days: int) -> dict:
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    return {
        **counts,
        "recall": recall,
        "fpr": fpr,
        "precision": tp / (tp + fp) if tp + fp else None,
        "youden_j": recall - fpr if recall is not None and fpr is not None else None,
        "false_alerts_per_user_week": fp * 7 / user_days,
        "alerts_per_user_week": counts["alerts"] * 7 / user_days,
        "user_days": user_days,
    }


def evaluate(rows: list[dict], config: PolicyConfig, days: int = 180) -> dict:
    """Контракт синтетики: ровно одно решение каждые 24 часа, полная траектория."""
    if not rows or days < 1:
        raise ValueError("Нужны непустые траектории и положительное число дней")
    groups = defaultdict(list)
    identities = set()
    for row in rows:
        key = (row["user_id"], row["event_id"])
        if key in identities:
            raise ValueError("Повтор идентификатора решения")
        identities.add(key)
        outcome = row["evaluation_outcome"]
        if outcome is not None and outcome not in (0, 1):
            raise ValueError("Исход должен быть 0, 1 или None")
        groups[row["user_id"]].append((datetime.fromisoformat(row["decision_at"]), row))
    people, decisions = {}, []
    for user, history in sorted(groups.items()):
        history.sort(key=lambda item: item[0])
        if len(history) != days or any(
            right[0] - left[0] != timedelta(days=1) for left, right in pairwise(history)
        ):
            raise ValueError("Нужны полные ежедневные траектории без пропусков решений")
        policy = AlertPolicy(config)
        counts = dict.fromkeys(
            ("tp", "fp", "tn", "fn", "unknown", "unknown_alerts", "alerts"), 0
        )
        for at, row in history:
            decision = policy.decide(at, float(row["p_final"]))
            # Исход доступен оценщику только после решения политики.
            outcome, alert = row["evaluation_outcome"], decision["alert"]
            counts["alerts"] += int(alert)
            if outcome is None:
                counts["unknown"] += 1
                counts["unknown_alerts"] += int(alert)
            else:
                key = (
                    ("tp" if alert else "fn")
                    if outcome == 1
                    else ("fp" if alert else "tn")
                )
                counts[key] += 1
            decisions.append(
                {
                    "user_id": user,
                    "event_id": row["event_id"],
                    "decision_at": row["decision_at"],
                    "p_final": row["p_final"],
                    **decision,
                    "evaluation_outcome": outcome,
                }
            )
        people[user] = _rates(counts, days)
    total = {key: sum(p[key] for p in people.values()) for key in counts}
    metrics = (
        "recall",
        "fpr",
        "precision",
        "youden_j",
        "false_alerts_per_user_week",
        "alerts_per_user_week",
    )
    valid = {
        key: [p[key] for p in people.values() if p[key] is not None] for key in metrics
    }
    return {
        "per_user": people,
        "decisions": decisions,
        "pooled": _rates(total, days * len(people)),
        "macro": {
            key: float(np.mean(values)) if values else None
            for key, values in valid.items()
        },
        "macro_users": {key: len(values) for key, values in valid.items()},
        "users": len(people),
        "decisions_count": len(rows),
        "label_scope": "all_outcomes"
        if total["unknown"] == 0
        else "observed_only_not_population_identified",
    }


def paired_intervals(
    baseline: dict, candidate: dict, draws: int = 10000, seed: int = 20261301
) -> dict:
    """Парный bootstrap пользователей внутри одного сида; только поисковые интервалы."""
    b, c = baseline["per_user"], candidate["per_user"]
    if set(b) != set(c):
        raise ValueError("Для парного сравнения требуются одинаковые пользователи")
    event_keys = ("user_id", "event_id", "decision_at", "evaluation_outcome")
    if [tuple(r[k] for k in event_keys) for r in baseline["decisions"]] != [
        tuple(r[k] for k in event_keys) for r in candidate["decisions"]
    ]:
        raise ValueError("Для парного сравнения требуются одинаковые события и исходы")
    rng, result = np.random.default_rng(seed), {}
    for key in ("recall", "fpr"):
        delta = np.array(
            [
                c[u][key] - b[u][key]
                for u in sorted(b)
                if c[u][key] is not None and b[u][key] is not None
            ]
        )
        ci = None
        if len(delta) >= 2:
            means = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(
                axis=1
            )
            ci = np.quantile(means, [0.025, 0.975]).tolist()
        result[key] = {
            "delta": float(delta.mean()) if len(delta) else None,
            "ci95": ci,
            "users": len(delta),
        }
    return {"contrasts": result, "draws": draws, "seed": seed, "status": "exploratory"}
