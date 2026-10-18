"""Модуль стресс-тестирования устойчивости и анализа чувствительности V12 (Этап 3).

Сценарии:
1. default: базовый режим (~75% отклика, задержка 20ч).
2. observability_50: ровно 50% всех решений получают обратную связь.
3. observability_10: ровно 10% всех решений получают обратную связь.
4. delay_72h: задержка обратной связи 72 часа.
5. delay_mixed: случайная экспоненциальная задержка (среднее 36ч).
6. label_noise_15: 15% случайной инверсии меток отчетов.
7. abrupt_drift: резкий скачок базового риска после 90-го дня.
8. temporary_drift: временный всплеск риска с 60 по 90 день.
9. mnar_forward: срывы репортятся чаще успехов (90% vs 35%).
10. mnar_reverse: срывы скрываются чаще успехов (30% vs 85%).
11. pseudolabel_impute_zero: запрещенная эвристика «нет ответа = успех (0)».
12. pseudolabel_impute_one: запрещенная эвристика «нет ответа = срыв (1)».
"""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from research_v11.metrics import compute_comprehensive_metrics
from research_v12.models import HierarchicalRidgeMemory, TunedScalarMemory


def transform_feedback_stream(
    records: list[dict],
    evaluations: list[dict],
    scenario: str,
    seed: int = 20261203,
) -> tuple[list[dict], list[dict]]:
    """Преобразует поток обратной связи согласно стресс-сценарию."""
    rng = np.random.default_rng(seed)
    eval_by_event = {e["event_id"]: e for e in evaluations}

    new_records = []
    new_evals = copy.deepcopy(evaluations)
    changed_evals = {e["event_id"]: e for e in new_evals}

    for r in records:
        r_new = copy.deepcopy(r)
        ev_id = r["event_id"]
        true_y = eval_by_event[ev_id]["evaluation_outcome"]
        t_dec = datetime.fromisoformat(r["decision_at"])

        if scenario == "default":
            pass

        elif scenario == "observability_50":
            # Независимая вероятность ответа 50%; фактическая доля сохраняется отдельно
            if rng.random() < 0.50:
                r_new["outcome"] = true_y
                r_new["feedback_at"] = (t_dec + timedelta(hours=20)).isoformat()
            else:
                r_new["outcome"] = None
                r_new["feedback_at"] = None

        elif scenario == "observability_10":
            # Независимая вероятность ответа 10%; это не квота на конечной выборке
            if rng.random() < 0.10:
                r_new["outcome"] = true_y
                r_new["feedback_at"] = (t_dec + timedelta(hours=20)).isoformat()
            else:
                r_new["outcome"] = None
                r_new["feedback_at"] = None

        elif scenario == "delay_72h":
            if r["outcome"] is not None:
                r_new["feedback_at"] = (t_dec + timedelta(hours=72)).isoformat()

        elif scenario == "delay_mixed":
            if r["outcome"] is not None:
                h = float(np.clip(rng.exponential(scale=36.0), 2.0, 120.0))
                r_new["feedback_at"] = (t_dec + timedelta(hours=h)).isoformat()

        elif scenario == "label_noise_15":
            if r["outcome"] is not None and rng.random() < 0.15:
                r_new["outcome"] = 1 - r_new["outcome"]

        elif scenario in {"abrupt_drift", "temporary_drift"}:
            # Сдвиг истинного риска при фиксированных признаках и маске репортинга.
            # Связанные исходы: каждый исход 0 становится 1 с вероятностью 0.4,
            # поэтому p_shift = p + 0.4 * (1 - p); вне окна исходы неизменны.
            info = eval_by_event[ev_id]
            day_idx = int(info["day_index"])
            active = day_idx >= 90 if scenario == "abrupt_drift" else 60 <= day_idx < 90
            if active:
                key = f"{seed}:{ev_id}:risk-drift".encode()
                event_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
                draw = np.random.default_rng(event_seed).random()
                shifted_y = int(true_y == 1 or draw < 0.40)
                p = float(info["evaluation_probability"])
                changed_evals[ev_id]["evaluation_probability"] = p + 0.40 * (1.0 - p)
                changed_evals[ev_id]["evaluation_outcome"] = shifted_y
                if r["outcome"] is not None:
                    r_new["outcome"] = shifted_y

        elif scenario == "mnar_forward":
            reported = rng.random() < (0.90 if true_y == 1 else 0.35)
            if reported:
                r_new["outcome"] = true_y
                r_new["feedback_at"] = (t_dec + timedelta(hours=20)).isoformat()
            else:
                r_new["outcome"] = None
                r_new["feedback_at"] = None

        elif scenario == "mnar_reverse":
            reported = rng.random() < (0.30 if true_y == 1 else 0.85)
            if reported:
                r_new["outcome"] = true_y
                r_new["feedback_at"] = (t_dec + timedelta(hours=20)).isoformat()
            else:
                r_new["outcome"] = None
                r_new["feedback_at"] = None

        elif scenario == "pseudolabel_impute_zero" and r["outcome"] is None:
            # Запрещенная эвристика: если ответа нет, подставляем y=0 через 24 часа
            r_new["outcome"] = 0
            r_new["feedback_at"] = (t_dec + timedelta(hours=24)).isoformat()

        elif scenario == "pseudolabel_impute_one" and r["outcome"] is None:
            # Запрещенная эвристика: если ответа нет, подставляем y=1 через 24 часа
            r_new["outcome"] = 1
            r_new["feedback_at"] = (t_dec + timedelta(hours=24)).isoformat()

        new_records.append(r_new)

    return new_records, new_evals


def run_scenario_evaluation(
    records: list[dict],
    evaluations: list[dict],
    p0: np.ndarray,
    contexts_z: np.ndarray,
    scalar_cfg: dict[str, Any],
    candidate_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Унифицированная оценка моделей с реальным измерением JSON-памяти."""
    eval_by_event = {e["event_id"]: e for e in evaluations}
    by_user: dict[str, list[tuple[int, dict]]] = {}
    for i, r in enumerate(records):
        by_user.setdefault(r["user_id"], []).append((i, r))

    scores_scalar = []
    scores_cand = []
    peak_state_bytes_scalar = 0
    peak_state_bytes_cand = 0

    dim = candidate_cfg.get("dimension", 8)

    for user_rows in by_user.values():
        user_rows.sort(key=lambda item: (item[1]["decision_at"], item[1]["event_id"]))
        mem_scalar = TunedScalarMemory(**scalar_cfg)
        mem_cand = HierarchicalRidgeMemory(**candidate_cfg)

        pending_scalar: list[tuple[datetime, str, int, float]] = []
        pending_cand: list[tuple[datetime, str, int, float]] = []

        feedbacks_scalar = 0
        feedbacks_cand = 0

        for idx, row in user_rows:
            t_dec = datetime.fromisoformat(row["decision_at"])
            ev_id = row["event_id"]
            p_pop = float(p0[idx])
            true_outcome = int(eval_by_event[ev_id]["evaluation_outcome"])
            z_vec = contexts_z[idx][:dim]

            # Обработка ответов скаляра
            pending_scalar.sort(key=lambda x: x[0])
            while pending_scalar and pending_scalar[0][0] < t_dec:
                fb_at, fb_ev_id, fb_outcome, fb_conf = pending_scalar.pop(0)
                mem_scalar.observe(fb_ev_id, fb_at, fb_outcome, fb_conf)
                feedbacks_scalar += 1
                peak_state_bytes_scalar = max(
                    peak_state_bytes_scalar, mem_scalar.state_size_bytes()
                )

            # Обработка ответов кандидата
            pending_cand.sort(key=lambda x: x[0])
            while pending_cand and pending_cand[0][0] < t_dec:
                fb_at, fb_ev_id, fb_outcome, fb_conf = pending_cand.pop(0)
                mem_cand.observe(fb_ev_id, fb_at, fb_outcome, fb_conf)
                feedbacks_cand += 1
                peak_state_bytes_cand = max(
                    peak_state_bytes_cand, mem_cand.state_size_bytes()
                )

            snap_s = mem_scalar.predict(ev_id, t_dec, p_pop)
            snap_c = mem_cand.predict(ev_id, t_dec, z_vec, p_pop)

            # Честное измерение размера памяти через сериализацию в JSON
            bytes_s = mem_scalar.state_size_bytes()
            bytes_c = mem_cand.state_size_bytes()
            peak_state_bytes_scalar = max(peak_state_bytes_scalar, bytes_s)
            peak_state_bytes_cand = max(peak_state_bytes_cand, bytes_c)

            scores_scalar.append(
                {
                    "user_id": row["user_id"],
                    "event_id": ev_id,
                    "p_population": p_pop,
                    "p_final": snap_s["p_final"],
                    "gamma": snap_s.get("gamma", 1.0),
                    "evaluation_outcome": true_outcome,
                    "feedbacks_processed": feedbacks_scalar,
                }
            )
            scores_cand.append(
                {
                    "user_id": row["user_id"],
                    "event_id": ev_id,
                    "p_population": p_pop,
                    "p_final": snap_c["p_final"],
                    "gamma": snap_c.get("gamma", 1.0),
                    "evaluation_outcome": true_outcome,
                    "feedbacks_processed": feedbacks_cand,
                }
            )

            # Добавление ожидающей обратной связи
            if row.get("feedback_at") and row.get("outcome") is not None:
                fb_time = datetime.fromisoformat(row["feedback_at"])
                pending_scalar.append(
                    (
                        fb_time,
                        ev_id,
                        int(row["outcome"]),
                        float(row.get("confidence", 1.0)),
                    )
                )
                pending_cand.append(
                    (
                        fb_time,
                        ev_id,
                        int(row["outcome"]),
                        float(row.get("confidence", 1.0)),
                    )
                )
            else:
                mem_scalar.discard(ev_id)
                mem_cand.discard(ev_id)

    metrics_s = compute_comprehensive_metrics(scores_scalar)
    metrics_c = compute_comprehensive_metrics(scores_cand)

    pop_brier = float(
        np.mean(
            [
                (float(s["p_population"]) - int(s["evaluation_outcome"])) ** 2
                for s in scores_scalar
            ]
        )
    )

    return {
        "pop_brier": pop_brier,
        "reported_fraction": sum(r["outcome"] is not None for r in records)
        / len(records),
        "decisions": len(records),
        "scalar_brier": metrics_s["macro_brier"],
        "cand_brier": metrics_c["macro_brier"],
        "diff_cand_minus_scalar": metrics_c["macro_brier"] - metrics_s["macro_brier"],
        "scalar_auc": metrics_s["macro_auc"],
        "cand_auc": metrics_c["macro_auc"],
        "scalar_logloss": metrics_s["macro_logloss"],
        "cand_logloss": metrics_c["macro_logloss"],
        "peak_state_bytes_scalar": peak_state_bytes_scalar,
        "peak_state_bytes_cand": peak_state_bytes_cand,
    }
