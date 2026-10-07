"""Причинный replay: одинаковые события, feedback и ограничения для всех правил."""

import heapq
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

from research_v7.types import MemoryConfig
from research_v10.memory import PlasticMemory
from research_v10.types import Variant


def replay(
    records: list[dict],
    codes: np.ndarray,
    p0: np.ndarray,
    variant: Variant,
    eligibility: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """Оценочные поля попадают в результат только после сохранения прогноза."""
    if len(records) != len(codes) or len(records) != len(p0):
        raise ValueError("Число строк не совпадает")
    if eligibility is None:
        eligibility = codes
    if eligibility.shape != codes.shape:
        raise ValueError("Размеры eligibility и кодировки не совпадают")
    users = defaultdict(list)
    for i, row in enumerate(records):
        users[row["user_id"]].append((i, row))
    result, sizes = [], []
    started = time.perf_counter()
    for user, history in sorted(users.items()):
        history.sort(key=lambda item: (item[1]["decision_at"], item[1]["event_id"]))
        config = MemoryConfig(fast_rate=variant.rate, slow_rate=variant.rate * 0.15)
        memory = (
            None
            if variant.family == "population"
            else PlasticMemory(codes.shape[1], variant.name, config, variant.rule)
        )
        pending = []
        post_change_seen = 0
        for day_index, (i, row) in enumerate(history):
            at = row["decision_at"]
            if row["available_at"] > at:
                raise ValueError("Будущие признаки")
            while pending and pending[0][0] < at:
                feedback_at, event_id, feedback, event_day = heapq.heappop(pending)
                if memory is not None:
                    memory.observe(
                        event_id,
                        feedback_at,
                        feedback["outcome"],
                        feedback.get("confidence", 1.0),
                    )
                if event_day >= row.get("change_day", 10**9):
                    post_change_seen += 1
            probability = candidate = raw = float(p0[i])
            gamma = 0.0
            if memory is not None:
                snap = memory.predict_with_trace(
                    row["event_id"], at, codes[i], float(p0[i]), eligibility[i]
                )
                probability, candidate, raw, gamma = (
                    snap.p_final,
                    snap.p_candidate,
                    snap.p_learning,
                    snap.gamma,
                )
            result.append(
                {
                    "user_id": user,
                    "event_id": row["event_id"],
                    "split": row["split"],
                    "decision_at": at.isoformat(),
                    "probability": probability,
                    "candidate": candidate,
                    "ungated": raw,
                    "p_population": float(p0[i]),
                    "gamma": gamma,
                    "outcome": row["evaluation_outcome"],
                    "observed_feedback": row["outcome"] is not None,
                    "true_probability": row.get("evaluation_probability"),
                    "day": day_index,
                    "change_day": row.get("change_day"),
                    "post_change_feedback": post_change_seen,
                }
            )
            if row["outcome"] is not None:
                if row["feedback_at"] is None or row["feedback_at"] <= at:
                    raise ValueError("Feedback обязан следовать за решением")
                heapq.heappush(
                    pending, (row["feedback_at"], row["event_id"], row, day_index)
                )
            elif memory is not None:
                memory.discard(row["event_id"])
        if memory is not None:
            sizes.append(
                {
                    "user_id": user,
                    "numeric_bytes": memory.fast.nbytes
                    + memory.slow.nbytes
                    + memory.exposure.nbytes,
                    "pending_trace_bytes": sum(
                        t.nbytes for t in memory.traces.values()
                    ),
                    "checkpoint_json_bytes": len(memory.checkpoint().encode()),
                    "pending": len(memory.pending),
                    "seen": len(memory.seen),
                    "observations": len(memory.observations),
                }
            )
    elapsed = time.perf_counter() - started
    return result, {
        "seconds": elapsed,
        "microseconds_per_decision": elapsed / max(1, len(records)) * 1e6,
        "states": sizes,
    }


def policy_rows(rows: list[dict], policy: str) -> list[dict]:
    if policy == "gated":
        return rows
    if policy != "ungated":
        raise ValueError("Неизвестная политика")
    return [{**row, "probability": row["ungated"]} for row in rows]


def parse_records(path) -> list[dict]:
    """Читает датированные события, не меняя observed/evaluation labels."""
    import json

    result = [json.loads(line) for line in path.read_text().splitlines()]
    for row in result:
        for key in ("decision_at", "available_at", "feedback_at"):
            row[key] = (
                datetime.fromisoformat(row[key]) if row.get(key) is not None else None
            )
    return result
