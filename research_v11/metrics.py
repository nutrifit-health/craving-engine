"""Расчёт метрик V11: Macro Brier, Log Loss, AUC, деградации и few-shot корзины."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Вычисляет средний бинарный log-loss с защитой от переполнения."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped)))


def compute_comprehensive_metrics(
    scores: list[dict],
    baseline_scores: list[dict] | None = None,
) -> dict[str, object]:
    """Вычисляет полный набор метрик для семейства прогнозов."""
    # Группировка по пользователям
    by_user_y: dict[str, list[int]] = {}
    by_user_p: dict[str, list[float]] = {}
    by_user_p_base: dict[str, list[float]] = {}

    base_p_map = {(r["user_id"], r["event_id"]): r["p_final"] for r in (baseline_scores or [])}

    for r in scores:
        u = r["user_id"]
        y = r["evaluation_outcome"]
        p = r["p_final"]
        by_user_y.setdefault(u, []).append(y)
        by_user_p.setdefault(u, []).append(p)
        if (u, r["event_id"]) in base_p_map:
            by_user_p_base.setdefault(u, []).append(base_p_map[(u, r["event_id"])])

    # 1. Macro Brier и Brier по пользователям
    user_briers = {
        u: float(np.mean((np.array(by_user_p[u]) - np.array(by_user_y[u])) ** 2))
        for u in by_user_y
    }
    macro_brier = float(np.mean(list(user_briers.values())))

    # 2. Macro Log Loss
    user_loglosses = {
        u: safe_log_loss(np.array(by_user_y[u]), np.array(by_user_p[u]))
        for u in by_user_y
    }
    macro_logloss = float(np.mean(list(user_loglosses.values())))

    # 3. Macro AUC & Pooled AUC
    all_y = np.concatenate([np.array(by_user_y[u]) for u in sorted(by_user_y)])
    all_p = np.concatenate([np.array(by_user_p[u]) for u in sorted(by_user_p)])
    pooled_auc = float(roc_auc_score(all_y, all_p)) if len(set(all_y)) == 2 else 0.5

    user_aucs = {}
    user_auc_deltas = {}
    for u, values in by_user_y.items():
        u_y = np.array(values)
        u_p = np.array(by_user_p[u])
        if len(set(u_y)) == 2:
            u_auc = float(roc_auc_score(u_y, u_p))
            user_aucs[u] = u_auc
            if u in by_user_p_base and len(set(u_y)) == 2:
                base_auc = float(roc_auc_score(u_y, np.array(by_user_p_base[u])))
                user_auc_deltas[u] = u_auc - base_auc

    macro_auc = float(np.mean(list(user_aucs.values()))) if user_aucs else 0.5
    evaluable_auc_users = len(user_aucs)
    total_users = len(by_user_y)

    degraded_count = sum(1 for delta in user_auc_deltas.values() if delta < -0.01)
    degraded_fraction = (degraded_count / len(user_auc_deltas)) if user_auc_deltas else 0.0

    # 4. Few-shot анализ по непересекающимся корзинам числа полученных ответов
    bins = [
        ("K=0", lambda k: k == 0),
        ("K=1-2", lambda k: 1 <= k <= 2),
        ("K=3-5", lambda k: 3 <= k <= 5),
        ("K=6-10", lambda k: 6 <= k <= 10),
        ("K=11+", lambda k: k >= 11),
    ]

    few_shot_results = {}
    for bin_name, bin_predicate in bins:
        bin_records = [r for r in scores if bin_predicate(r["feedbacks_processed"])]
        if not bin_records:
            few_shot_results[bin_name] = {
                "n_decisions": 0,
                "n_users": 0,
                "gate_open_fraction": 0.0,
                "brier": None,
                "log_loss": None,
            }
            continue

        bin_y = np.array([r["evaluation_outcome"] for r in bin_records], dtype=np.float64)
        bin_p = np.array([r["p_final"] for r in bin_records], dtype=np.float64)
        gate_open = sum(1 for r in bin_records if r["gamma"] > 0)
        active_users = len({r["user_id"] for r in bin_records})

        few_shot_results[bin_name] = {
            "n_decisions": len(bin_records),
            "n_users": active_users,
            "gate_open_fraction": float(gate_open / len(bin_records)),
            "brier": float(np.mean((bin_p - bin_y) ** 2)),
            "log_loss": safe_log_loss(bin_y, bin_p),
        }

    user_logloss = {}
    for u, values in by_user_y.items():
        u_y = np.array(values, dtype=np.float64)
        u_p = np.array(by_user_p[u], dtype=np.float64)
        user_logloss[u] = safe_log_loss(u_y, u_p)

    return {
        "macro_brier": macro_brier,
        "user_briers": user_briers,
        "macro_logloss": macro_logloss,
        "user_logloss": user_logloss,
        "macro_auc": macro_auc,
        "user_aucs": user_aucs,
        "user_auc_deltas": user_auc_deltas,
        "pooled_auc": pooled_auc,
        "evaluable_auc_users": evaluable_auc_users,
        "total_users": total_users,
        "degraded_auc_users_count": degraded_count,
        "degraded_auc_users_fraction": degraded_fraction,
        "few_shot": few_shot_results,
    }
