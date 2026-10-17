"""Причинный replay для когорт V11 с отслеживанием few-shot и пикового размера состояния."""

from __future__ import annotations

import heapq
from collections import defaultdict
from datetime import datetime
from typing import Literal

import numpy as np

from research_v11.memory import CompactMemoryV11
from research_v11.types import MemoryConfigV11


def run_v11_replay(
    records: list[dict],
    evaluations: list[dict],
    p_population: np.ndarray,
    contexts_z: np.ndarray | None,
    family: Literal["population", "scalar", "compact_linear", "compact_kc_anatomical", "compact_kc_rewired"],
    dimension: int = 0,
    config: MemoryConfigV11 | None = None,
) -> tuple[list[dict], dict[str, object]]:
    """Проводит хронологический причинный replay для когорты пользователей."""
    cfg = config or MemoryConfigV11()
    eval_by_event = {e["event_id"]: e for e in evaluations}

    # Группировка по пользователям
    by_user = defaultdict(list)
    for i, r in enumerate(records):
        by_user[r["user_id"]].append((i, r))

    scores = []
    user_peak_bytes = {}
    user_feedbacks_count = {}

    for user_id, user_rows in sorted(by_user.items()):
        # Хронологическая сортировка
        user_rows.sort(key=lambda item: (item[1]["decision_at"], item[1]["event_id"]))

        memory = None
        if family != "population":
            dim = 0 if family == "scalar" else dimension
            memory = CompactMemoryV11(dimension=dim, config=cfg)

        pending_feedback: list[tuple[datetime, str, int, float]] = []
        feedbacks_processed = 0
        peak_bytes = 0

        for idx, row in user_rows:
            dec_at = datetime.fromisoformat(row["decision_at"])
            event_id = row["event_id"]
            p_pop = float(p_population[idx])
            eval_info = eval_by_event[event_id]
            true_outcome = int(eval_info["evaluation_outcome"])

            # Причинная обработка поступивших ответов строго до момента решения
            while pending_feedback and pending_feedback[0][0] < dec_at:
                fb_at, fb_ev_id, fb_outcome, fb_conf = heapq.heappop(pending_feedback)
                if memory is not None:
                    memory.observe(fb_ev_id, fb_at, fb_outcome, fb_conf)
                    peak_bytes = max(peak_bytes, memory.state_size_bytes())
                feedbacks_processed += 1

            p_final = p_pop
            p_learning = p_pop
            gamma = 0.0
            familiarity = 0.0
            residual = 0.0
            b_res = 0.0
            w_res = 0.0

            if memory is not None:
                z_vec = None
                if family != "scalar" and contexts_z is not None:
                    z_vec = contexts_z[idx]
                snap = memory.predict(event_id, dec_at, z_vec, p_pop)
                p_final = snap.p_final
                p_learning = snap.p_learning
                gamma = snap.gamma
                familiarity = snap.familiarity
                residual = snap.residual
                b_res = snap.b_residual
                w_res = snap.w_residual
                peak_bytes = max(peak_bytes, memory.state_size_bytes())

            # Запись результата решения
            scores.append({
                "family": family,
                "user_id": user_id,
                "event_id": event_id,
                "decision_at": row["decision_at"],
                "p_population": p_pop,
                "p_final": p_final,
                "p_learning": p_learning,
                "gamma": gamma,
                "familiarity": familiarity,
                "residual": residual,
                "b_residual": b_res,
                "w_residual": w_res,
                "evaluation_outcome": true_outcome,
                "feedbacks_processed": feedbacks_processed,
            })

            # Регистрация ответа в очередь задержки или удаление снимка
            if row["outcome"] is not None and row["feedback_at"] is not None:
                fb_time = datetime.fromisoformat(row["feedback_at"])
                conf = float(row.get("confidence", 1.0))
                heapq.heappush(pending_feedback, (fb_time, event_id, int(row["outcome"]), conf))
            elif memory is not None:
                memory.discard(event_id)

        user_peak_bytes[user_id] = peak_bytes
        user_feedbacks_count[user_id] = feedbacks_processed

    diagnostics = {
        "family": family,
        "dimension": dimension,
        "n_users": len(by_user),
        "n_decisions": len(scores),
        "peak_state_bytes_max": max(user_peak_bytes.values()) if user_peak_bytes else 0,
        "peak_state_bytes_mean": float(np.mean(list(user_peak_bytes.values()))) if user_peak_bytes else 0.0,
        "feedbacks_mean": float(np.mean(list(user_feedbacks_count.values()))) if user_feedbacks_count else 0.0,
    }
    return scores, diagnostics
