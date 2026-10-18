"""Модуль диагностики V12: исследование причин поражения V11, абляции и контрольные бенчмарки.

Реализует:
1. Абляции механизмов: гейт (gamma), familiarity, clipping логитов.
2. Анализ информативности PCA-представления vs residual signal (Вопрос D).
3. Анализ внутрипользовательской предсказуемости (Вопрос B).
4. Два синтетических бенчмарка: Intercept-only и Known contextual effect.
5. Оракул-диагностику (True probability, Retrospective fit, Privileged profile).
"""

from __future__ import annotations

import copy
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from hybrid_engine import safe_logit, safe_sigmoid
from research_v11.memory import CompactMemoryV11
from research_v11.metrics import compute_comprehensive_metrics
from research_v11.types import MemoryConfigV11


def run_ablation_experiment(
    records: list[dict],
    evaluations: list[dict],
    p_pop: np.ndarray,
    contexts_z: np.ndarray | None,
    dimension: int,
    base_config: MemoryConfigV11,
    ablation_type: str,
) -> dict[str, Any]:
    """Запускает причинный replay с определенной абляцией."""
    cfg = copy.deepcopy(base_config)
    eval_by_event = {e["event_id"]: e for e in evaluations}

    by_user: dict[str, list[tuple[int, dict]]] = {}
    for i, r in enumerate(records):
        by_user.setdefault(r["user_id"], []).append((i, r))

    scores = []
    gammas = []
    familiarities = []
    residuals = []
    clipped_count = 0
    total_predictions = 0

    for user_id, user_rows in sorted(by_user.items()):
        user_rows.sort(key=lambda item: (item[1]["decision_at"], item[1]["event_id"]))
        memory = CompactMemoryV11(dimension=dimension, config=cfg)
        pending: list[tuple[datetime, str, int, float]] = []

        feedbacks_processed = 0
        for idx, row in user_rows:
            dec_at = datetime.fromisoformat(row["decision_at"])
            event_id = row["event_id"]
            p0 = float(p_pop[idx])
            eval_info = eval_by_event[event_id]
            true_outcome = int(eval_info["evaluation_outcome"])

            # Feedback processing
            pending.sort(key=lambda x: x[0])
            while pending and pending[0][0] < dec_at:
                fb_at, fb_ev_id, fb_outcome, fb_conf = pending.pop(0)
                memory.observe(fb_ev_id, fb_at, fb_outcome, fb_conf)
                feedbacks_processed += 1

            z_vec = (
                contexts_z[idx]
                if contexts_z is not None and dimension > 0
                else np.zeros(0)
            )

            if ablation_type == "no_clipping":
                cfg = cfg.model_copy(update={"max_logit_residual": 1e9})
                memory.config = cfg

            snap = memory.predict(
                event_id, dec_at, z_vec if dimension > 0 else None, p0
            )

            if abs(snap.residual) >= base_config.max_logit_residual:
                clipped_count += 1
            total_predictions += 1

            gamma = snap.gamma
            familiarity = snap.familiarity

            if ablation_type in ("no_familiarity", "no_gate_no_familiarity"):
                familiarity = 1.0
            if ablation_type in ("no_gate", "no_gate_no_familiarity"):
                gamma = 1.0

            base = safe_logit(p0, eps=1e-12)
            candidate_residual = familiarity * snap.residual
            learning = snap.p_learning
            candidate = safe_sigmoid(base + candidate_residual)
            final = (
                p0
                if gamma == 0 or snap.residual == 0
                else safe_sigmoid(base + gamma * familiarity * snap.residual)
            )

            snap_ablated = snap.model_copy(
                update={
                    "p_final": final,
                    "p_candidate": candidate,
                    "gamma": gamma,
                    "familiarity": familiarity,
                }
            )
            memory.pending[event_id] = snap_ablated

            scores.append(
                {
                    "user_id": user_id,
                    "event_id": event_id,
                    "decision_at": row["decision_at"],
                    "p_population": p0,
                    "p_final": final,
                    "p_learning": learning,
                    "gamma": gamma,
                    "familiarity": familiarity,
                    "residual": snap.residual,
                    "evaluation_outcome": true_outcome,
                    "feedbacks_processed": feedbacks_processed,
                }
            )
            gammas.append(gamma)
            familiarities.append(familiarity)
            residuals.append(snap.residual)

            if row.get("feedback_at") and row.get("outcome") is not None:
                fb_at = datetime.fromisoformat(row["feedback_at"])
                pending.append(
                    (
                        fb_at,
                        event_id,
                        int(row["outcome"]),
                        float(row.get("confidence", 1.0)),
                    )
                )
            else:
                memory.discard(event_id)

    metrics = compute_comprehensive_metrics(scores)
    return {
        "ablation": ablation_type,
        "dimension": dimension,
        "metrics": metrics,
        "zero_gamma_pct": float(np.mean(np.array(gammas) == 0.0) * 100),
        "mean_gamma": float(np.mean(gammas)),
        "mean_familiarity": float(np.mean(familiarities)),
        "clipped_pct": float(clipped_count / max(1, total_predictions) * 100),
        "mean_abs_residual": float(np.mean(np.abs(residuals))),
    }


def run_question_d_pca_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    p0_train: np.ndarray,
) -> dict[str, Any]:
    """Количественный анализ Вопроса D: PCA vs residual signal."""
    residual = y_train - p0_train
    mean = np.mean(X_train, axis=0)
    std = np.where(np.std(X_train, axis=0) > 1e-4, np.std(X_train, axis=0), 1.0)
    X_norm = (X_train - mean) / std

    pca = PCA(n_components=8, random_state=20261201).fit(X_norm)
    Z_pca = pca.transform(X_norm)

    ridge_pca = Ridge(alpha=100.0).fit(Z_pca, residual)
    r2_pca = float(ridge_pca.score(Z_pca, residual))

    ridge_raw = Ridge(alpha=100.0).fit(X_norm, residual)
    r2_raw = float(ridge_raw.score(X_norm, residual))

    corrs = [float(np.corrcoef(Z_pca[:, i], residual)[0, 1]) for i in range(8)]
    return {
        "pca_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "total_pca_variance_8": float(np.sum(pca.explained_variance_ratio_)),
        "residual_r2_pca": r2_pca,
        "residual_r2_raw_features": r2_raw,
        "pca_component_residual_correlations": corrs,
    }


def generate_intercept_only_benchmark(
    n_users: int = 40,
    seed: int = 20261208,
) -> tuple[np.ndarray, list[dict], list[dict], np.ndarray]:
    """Синтетический отрицательный контроль: варьирует только willpower intercept b_u, w_u = 0."""
    rng = np.random.default_rng(seed)
    n_days = 180
    n_features = 8

    # Популяционные коэффициенты одинаковы для всех пользователей
    beta_pop = rng.normal(0, 0.5, size=n_features)

    records = []
    evaluations = []
    X_list = []
    p0_list = []

    start = datetime(2026, 1, 1, tzinfo=UTC)

    for u_idx in range(n_users):
        u_id = f"V12-INTO-{u_idx + 1:03d}"
        # Индивидуальный intercept
        b_u = float(rng.normal(0.0, 0.8))
        # Контекстные чувствительности идентичны нулю для всех пользователей!
        # w_u = 0

        for day in range(n_days):
            t_dec = start + timedelta(days=day)
            event_id = f"{u_id}:{day}"
            x = rng.normal(0, 1.0, size=n_features)
            X_list.append(x)

            # Истинный логит: beta_pop @ x + b_u
            true_logit = float(beta_pop @ x + b_u)
            prob = 1.0 / (1.0 + math.exp(-true_logit))
            outcome = int(rng.random() < prob)

            # Популяционный прогноз p0: использует только beta_pop @ x (без знания b_u)
            pop_logit = float(beta_pop @ x)
            p0 = 1.0 / (1.0 + math.exp(-pop_logit))
            p0_list.append(p0)

            reported = rng.random() < 0.75
            feedback_at = (
                (t_dec + timedelta(hours=20)).isoformat() if reported else None
            )

            records.append(
                {
                    "user_id": u_id,
                    "event_id": event_id,
                    "decision_at": t_dec.isoformat(),
                    "available_at": t_dec.isoformat(),
                    "outcome": outcome if reported else None,
                    "feedback_at": feedback_at,
                    "confidence": 1.0,
                }
            )
            evaluations.append(
                {
                    "user_id": u_id,
                    "event_id": event_id,
                    "day_index": day,
                    "decision_at": t_dec.isoformat(),
                    "evaluation_outcome": outcome,
                    "evaluation_probability": prob,
                }
            )

    X = np.asarray(X_list, dtype=np.float64)
    p0 = np.asarray(p0_list, dtype=np.float64)
    return X, records, evaluations, p0


def generate_known_contextual_benchmark(
    n_users: int = 40,
    seed: int = 20261209,
    effect_magnitude: float = 1.0,
) -> tuple[np.ndarray, list[dict], list[dict], np.ndarray]:
    """Синтетический положительный контроль: внедрен истинный индивидуальный наклон w_u."""
    rng = np.random.default_rng(seed)
    n_days = 180
    n_features = 8

    beta_pop = rng.normal(0, 0.3, size=n_features)
    records = []
    evaluations = []
    X_list = []
    p0_list = []

    start = datetime(2026, 1, 1, tzinfo=UTC)

    for u_idx in range(n_users):
        u_id = f"V12-KNOW-{u_idx + 1:03d}"
        b_u = float(rng.normal(0.0, 0.5))
        # Индивидуальный наклон w_u фиксированной амплитуды
        w_u = rng.normal(0.0, effect_magnitude, size=n_features)

        for day in range(n_days):
            t_dec = start + timedelta(days=day)
            event_id = f"{u_id}:{day}"
            x = rng.normal(0, 1.0, size=n_features)
            X_list.append(x)

            # Истинный логит включает индивидуальный w_u @ x
            true_logit = float((beta_pop + w_u) @ x + b_u)
            prob = 1.0 / (1.0 + math.exp(-true_logit))
            outcome = int(rng.random() < prob)

            # Популяционный прогноз p0
            pop_logit = float(beta_pop @ x)
            p0 = 1.0 / (1.0 + math.exp(-pop_logit))
            p0_list.append(p0)

            reported = rng.random() < 0.75
            feedback_at = (
                (t_dec + timedelta(hours=20)).isoformat() if reported else None
            )

            records.append(
                {
                    "user_id": u_id,
                    "event_id": event_id,
                    "decision_at": t_dec.isoformat(),
                    "available_at": t_dec.isoformat(),
                    "outcome": outcome if reported else None,
                    "feedback_at": feedback_at,
                    "confidence": 1.0,
                }
            )
            evaluations.append(
                {
                    "user_id": u_id,
                    "event_id": event_id,
                    "day_index": day,
                    "decision_at": t_dec.isoformat(),
                    "evaluation_outcome": outcome,
                    "evaluation_probability": prob,
                }
            )

    X = np.asarray(X_list, dtype=np.float64)
    p0 = np.asarray(p0_list, dtype=np.float64)
    return X, records, evaluations, p0
