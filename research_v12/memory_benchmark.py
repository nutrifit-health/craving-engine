"""Измерение пиков полного checkpoint на одном причинном потоке, без обучения моделей."""

from __future__ import annotations

import heapq
from datetime import UTC, datetime, timedelta

import numpy as np

from research_v12.models import HierarchicalRidgeMemory, MoEMemory, TunedScalarMemory


def measure_memory_stream(memory, *, days: int = 365, seed: int = 20261201) -> dict:
    """Замеряет каждый predict/observe/expire; нерепорты хранятся до TTL 7 дней.

    75% ответов; задержки 20/72/120 ч. Одинаковая траектория для всех методов.
    После последнего дня очередь дренируется, эти замеры выделены отдельно.
    Это нагрузочный синтетический benchmark состояния, а не оценка качества.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    events = []
    measurements = []
    max_pending = 0

    def record(at, phase):
        nonlocal max_pending
        size = memory.state_size_bytes()
        max_pending = max(max_pending, len(memory.pending))
        measurements.append({"at": at.isoformat(), "phase": phase, "bytes": size})

    def drain(until):
        while events and events[0][0] < until:
            at, event_id, outcome = heapq.heappop(events)
            if outcome is None:
                memory.discard(event_id)
            else:
                memory.observe(event_id, at, outcome)
            record(at, "expire" if outcome is None else "observe")

    for day in range(days):
        at = start + timedelta(days=day)
        drain(at)
        event_id = f"memory-load-{day:04d}"
        p0 = float(np.clip(rng.normal(0.22, 0.05), 0.05, 0.70))
        z = rng.normal(size=8)
        if isinstance(memory, HierarchicalRidgeMemory):
            memory.predict(event_id, at, z[: memory.dimension], p0)
        elif isinstance(memory, MoEMemory):
            memory.predict(event_id, at, np.full(memory.k_experts, p0))
        elif isinstance(memory, TunedScalarMemory):
            memory.predict(event_id, at, p0)
        else:
            raise TypeError("Неизвестная модель памяти")
        record(at, "predict")
        reported = rng.random() < 0.75
        delay = int(rng.choice([20, 72, 120])) if reported else 168
        outcome = int(rng.random() < p0) if reported else None
        heapq.heappush(events, (at + timedelta(hours=delay), event_id, outcome))

    drain(start + timedelta(days=days))
    trajectory = list(measurements)
    drain(start + timedelta(days=days + 8))
    peaks = {}
    for horizon in (180, 365):
        if horizon <= days:
            end = start + timedelta(days=horizon)
            eligible = [m for m in trajectory if datetime.fromisoformat(m["at"]) < end]
            peak = max(eligible, key=lambda m: m["bytes"])
            peaks[f"peak_bytes_{horizon}_days"] = peak["bytes"]
            peaks[f"peak_event_{horizon}_days"] = peak
    overall = max(m["bytes"] for m in measurements)
    return {
        **peaks,
        "peak_including_drain": overall,
        "max_pending": max_pending,
        "budget_bytes": 16384,
        "within_budget": overall <= 16384,
        "days": days,
        "seed": seed,
        "config": memory.get_config(),
        "measurement_method": "полный JSON после каждого predict/observe/expire",
        "measurements": measurements,
    }
