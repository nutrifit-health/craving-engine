"""Отделяет обнаружение исхода от причинного ограничения уведомлений."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta

import numpy as np

from research_v6.replay import bootstrap_interval
from synthetic_benchmark.metrics import compute_roc_auc


def metrics(rows: list[dict], threshold: float, training_prevalence: float) -> dict:
    """Скрытый исход используется только оценщиком уже сохранённых прогнозов."""
    scored = [row for row in rows if row["outcome"] is not None]
    y = np.asarray([row["outcome"] for row in scored], dtype=float)
    p = np.asarray([row["probability"] for row in scored], dtype=float)
    candidate = np.asarray([row["candidate"] for row in scored], dtype=float)
    detected = p >= threshold
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    tp = int(np.sum(detected & (y == 1)))
    fp = int(np.sum(detected & (y == 0)))
    brier = float(np.mean((p - y) ** 2)) if len(y) else None
    constant_brier = float(np.mean((training_prevalence - y) ** 2)) if len(y) else None
    return {
        "decisions": len(rows), "scored": len(y), "events": positives,
        "observed_feedback": sum(row["observed_feedback"] for row in rows),
        "brier": brier,
        "candidate_brier": float(np.mean((candidate - y) ** 2)) if len(y) else None,
        "bss_training_prevalence": 1 - brier / constant_brier if constant_brier else None,
        "auc": compute_roc_auc(y.tolist(), p.tolist()) if positives and negatives else None,
        "candidate_auc": compute_roc_auc(y.tolist(), candidate.tolist()) if positives and negatives else None,
        "threshold": threshold, "true_positives": tp, "false_positives": fp,
        "recall": tp / positives if positives else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "false_positive_rate": fp / negatives if negatives else None,
        "false_alarms_per_caught": fp / tp if tp else None,
        "mean_gamma": float(np.mean([row["gamma"] for row in rows])) if rows else 0,
        "users_with_open_gate": len({row["user_id"] for row in rows if row["gamma"] > 0}),
    }


def choose_thresholds(calibration_rows: list[dict]) -> dict[str, float]:
    """Пороги определяются только на calibration; evaluation сюда не передаётся."""
    if not calibration_rows or any(row["split"] != "calibration" for row in calibration_rows):
        raise ValueError("Выбор порогов требует непустой calibration-выборки")
    scored = [row for row in calibration_rows if row["outcome"] is not None]
    y = np.asarray([row["outcome"] for row in scored])
    p = np.asarray([row["probability"] for row in scored])
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Calibration должна содержать оба исхода")
    grid = []
    for threshold in np.linspace(0, 1, 201):
        detected = p >= threshold
        recall = float(np.mean(detected[y == 1]))
        fpr = float(np.mean(detected[y == 0]))
        grid.append((float(threshold), recall, fpr))
    # При одинаковом recall выбирается меньший FPR, затем более высокий порог.
    moderate = max((row for row in grid if row[2] <= 0.2), key=lambda row: (row[1], -row[2], row[0]))
    sensitive = max((row for row in grid if row[1] >= 0.94), key=lambda row: row[0])
    return {"fixed_030": 0.3, "calibration_fpr20": moderate[0], "calibration_recall94": sensitive[0]}


def notification_metrics(rows: list[dict], threshold: float, limit: int) -> dict:
    """Пуш определяется до доступа к исходу; лимит общий для всех целей человека."""
    if limit < 1:
        raise ValueError("Лимит должен быть положительным")
    queues: dict[str, deque] = defaultdict(deque)
    sent = []
    for row in sorted(rows, key=lambda item: (item["decision_at"], item["user_id"], item["event_id"])):
        at = datetime.fromisoformat(row["decision_at"])
        queue = queues[row["user_id"]]
        while queue and at - queue[0] >= timedelta(days=7):
            queue.popleft()
        notify = row["probability"] >= threshold and len(queue) < limit
        if notify:
            queue.append(at)
            sent.append(row)
    events = sum(row["outcome"] == 1 for row in rows)
    detected = sum(row["outcome"] == 1 and row["probability"] >= threshold for row in rows)
    tp = sum(row["outcome"] == 1 for row in sent)
    fp = sum(row["outcome"] == 0 for row in sent)
    return {
        "rolling_seven_day_limit": limit, "threshold": threshold,
        "notifications": len(sent), "caught_events": tp, "false_notifications": fp,
        "events_missed_by_classifier": events - detected,
        "detected_events_blocked_by_limit": detected - tp,
        "recall": tp / events if events else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "false_alarms_per_caught": fp / tp if tp else None,
    }


def model_report(rows: list[dict], thresholds: dict[str, float], training_prevalence: float) -> dict:
    by_user: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_user[row["user_id"]].append(row)
    per_user = {user: metrics(items, 0.3, training_prevalence) for user, items in by_user.items()}
    for user, items in by_user.items():
        notifications = notification_metrics(items, 0.3, 2)
        # Достаточные счётчики для парного bootstrap без сохранения его выборок.
        per_user[user]["fixed_030_notifications_limit2"] = {
            key: notifications[key]
            for key in ("caught_events", "notifications", "false_notifications")
        }
    aucs = [item["auc"] for item in per_user.values() if item["auc"] is not None]
    return {
        "pooled": metrics(rows, 0.3, training_prevalence), "per_user": per_user,
        "macro_user_auc": float(np.mean(aucs)) if aucs else None,
        "operating_points": {
            name: {
                "detection": metrics(rows, threshold, training_prevalence),
                "notifications": {str(limit): notification_metrics(rows, threshold, limit) for limit in (2, 3, 7)},
            } for name, threshold in thresholds.items()
        },
    }


def _paired_recall_bootstrap(
    candidate_detected: list[int],
    baseline_detected: list[int],
    candidate_notified: list[int],
    baseline_notified: list[int],
    events: list[int],
    seed: int,
    draws: int = 2000,
) -> dict:
    """Пересчитывает pooled TP/events на одинаковых кластерах обеих моделей."""
    total_events = sum(events)
    result = {
        "threshold": 0.3, "rolling_seven_day_notification_limit": 2,
        "users": len(events), "bootstrap_requested_draws": draws,
        "bootstrap_valid_draws": 0, "bootstrap_undefined_draws": 0,
        "pooled_recall_gain": (
            (sum(candidate_detected) - sum(baseline_detected)) / total_events
            if total_events else None
        ),
        "pooled_recall_gain_user_bootstrap_95": None,
        "notified_recall_gain": (
            (sum(candidate_notified) - sum(baseline_notified)) / total_events
            if total_events else None
        ),
        "notified_recall_gain_user_bootstrap_95": None,
        "bootstrap_status": "ok",
    }
    if len(events) < 2 or not total_events:
        result["bootstrap_status"] = "insufficient_users" if len(events) < 2 else "no_positive_events"
        return result
    counts = np.asarray([
        candidate_detected, baseline_detected, candidate_notified, baseline_notified, events,
    ], dtype=np.int64).T
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(events), size=(draws, len(events)))
    sampled = counts[indices].sum(axis=1)
    valid = sampled[:, 4] > 0
    result["bootstrap_valid_draws"] = int(valid.sum())
    result["bootstrap_undefined_draws"] = int((~valid).sum())
    if valid.sum() < 2:
        result["bootstrap_status"] = "insufficient_defined_draws"
        return result
    sampled = sampled[valid]
    detection_gains = (sampled[:, 0] - sampled[:, 1]) / sampled[:, 4]
    notification_gains = (sampled[:, 2] - sampled[:, 3]) / sampled[:, 4]
    result["pooled_recall_gain_user_bootstrap_95"] = np.quantile(detection_gains, [0.025, 0.975]).tolist()
    result["notified_recall_gain_user_bootstrap_95"] = np.quantile(notification_gains, [0.025, 0.975]).tolist()
    if np.any(~valid):
        result["bootstrap_status"] = "undefined_no_event_draws_excluded"
    return result


def paired_comparison(candidate: dict, baseline: dict, seed: int = 20260923) -> dict:
    """Единица повторной выборки — пользователь, а не зависимые дни."""
    if set(candidate["per_user"]) != set(baseline["per_user"]):
        raise ValueError("Парное сравнение требует одинаковый состав пользователей")
    brier, candidate_brier, auc = [], [], []
    candidate_detected, baseline_detected, candidate_notified, baseline_notified, events = [], [], [], [], []
    for user in sorted(candidate["per_user"]):
        item = candidate["per_user"][user]
        reference = baseline["per_user"][user]
        if item["events"] != reference["events"] or item["scored"] != reference["scored"]:
            raise ValueError("Парное сравнение требует одинаковые оценочные исходы и покрытие")
        if item["threshold"] != 0.3 or reference["threshold"] != 0.3:
            raise ValueError("Диагностика recall требует фиксированный порог 0.30 у обеих моделей")
        if item["brier"] is not None and reference["brier"] is not None:
            brier.append(reference["brier"] - item["brier"])
        if item["candidate_brier"] is not None and reference["candidate_brier"] is not None:
            candidate_brier.append(reference["candidate_brier"] - item["candidate_brier"])
        if item["auc"] is not None and reference["auc"] is not None:
            auc.append(item["auc"] - reference["auc"])
        events.append(item["events"])
        candidate_detected.append(item["true_positives"])
        baseline_detected.append(reference["true_positives"])
        candidate_notified.append(item["fixed_030_notifications_limit2"]["caught_events"])
        baseline_notified.append(reference["fixed_030_notifications_limit2"]["caught_events"])
    return {
        "bootstrap_seed": seed, "bootstrap_draws": 2000, "bootstrap_unit": "user",
        "mean_brier_gain": float(np.mean(brier)) if brier else None,
        "brier_gain_user_bootstrap_95": bootstrap_interval(brier, seed, 2000),
        "mean_candidate_brier_gain": float(np.mean(candidate_brier)) if candidate_brier else None,
        "candidate_brier_gain_user_bootstrap_95": bootstrap_interval(candidate_brier, seed, 2000),
        "mean_auc_gain": float(np.mean(auc)) if auc else None,
        "auc_gain_user_bootstrap_95": bootstrap_interval(auc, seed, 2000),
        "fraction_auc_degraded_over_001": sum(value < -0.01 for value in auc) / len(auc) if auc else None,
        "fixed_030_recall_comparison": _paired_recall_bootstrap(
            candidate_detected, baseline_detected, candidate_notified, baseline_notified,
            events, seed,
        ),
    }
