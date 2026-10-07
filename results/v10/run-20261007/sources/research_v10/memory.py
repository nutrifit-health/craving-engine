"""Readout с сохранённой eligibility и неизменёнными ограничениями V9."""

import json
from datetime import datetime

import numpy as np

from research_v7.types import MemoryConfig
from research_v8.memory import EncodedMemory
from research_v8.types import Observation, Snapshot


class PlasticMemory(EncodedMemory):
    """Общий baseline и гейт; меняется только направление локального обновления."""

    def __init__(
        self, dimension: int, identity: str, config: MemoryConfig, rule: str = "direct"
    ):
        super().__init__(dimension, identity, config)
        if rule not in ("direct", "stdp", "rstdp"):
            raise ValueError("Неизвестное правило пластичности")
        self.rule = rule
        self.traces: dict[str, np.ndarray] = {}

    def predict_with_trace(
        self,
        event_id: str,
        at: datetime,
        code: np.ndarray,
        p0: float,
        eligibility: np.ndarray,
    ) -> Snapshot:
        if (
            eligibility.shape != (self.dimension,)
            or not np.isfinite(eligibility).all()
            or not np.isclose(np.linalg.norm(eligibility), 1)
        ):
            raise ValueError("Eligibility должна иметь единичную норму")
        snap = self.predict(event_id, at, code, p0)
        self.traces[event_id] = eligibility.copy()
        return snap

    def observe(
        self, event_id: str, at: datetime, outcome: int, confidence: float = 1.0
    ) -> None:
        if self.rule == "direct":
            super().observe(event_id, at, outcome, confidence)
            self.traces.pop(event_id, None)
            return
        if event_id not in self.pending or event_id not in self.traces:
            raise ValueError("Неизвестный либо повторный feedback")
        snap = self.pending[event_id]
        if (
            outcome not in (0, 1)
            or not 0 < confidence <= 1
            or at.utcoffset() is None
            or at <= snap.decision_at
        ):
            raise ValueError("Некорректный feedback")
        self._advance(at)
        indices = np.asarray(snap.indices, dtype=int)
        experience = float(np.mean(self.exposure[indices])) if len(indices) else 0.0
        consolidation = experience / (experience + 3)
        age = (at - snap.decision_at).total_seconds() / 86400
        # Без модулятора STDP игнорирует значение метки, но использует ту же
        # доступность feedback. Это контроль, не самостоятельный предиктор цели.
        signal = (outcome - snap.p_learning) if self.rule == "rstdp" else 1.0
        self.fast *= 1 - min(1.0, self.config.fast_rate * self.config.l2 * confidence)
        self.slow *= 1 - min(1.0, self.config.slow_rate * self.config.l2 * confidence)
        delta = signal * confidence * self.traces[event_id]
        self.fast += (
            self.config.fast_rate
            * 2 ** (-age / self.config.fast_half_life_days)
            * delta
        )
        self.slow += (
            self.config.slow_rate
            * 2 ** (-age / self.config.slow_half_life_days)
            * consolidation
            * delta
        )
        self.exposure[indices] += confidence * 2 ** (
            -age / self.config.exposure_half_life_days
        )
        self.observations.append(
            Observation(
                decision_at=snap.decision_at,
                feedback_at=at,
                p_population=snap.p_population,
                p_candidate=snap.p_candidate,
                outcome=outcome,
                confidence=confidence,
            )
        )
        del self.pending[event_id], self.traces[event_id]

    def discard(self, event_id: str) -> None:
        super().discard(event_id)
        self.traces.pop(event_id, None)

    def checkpoint(self) -> str:
        """Размер включает pending, seen, историю гейта и eligibility."""
        return json.dumps(
            {
                "version": 1,
                "rule": self.rule,
                "base": json.loads(self.export_state()),
                "traces": {key: value.tolist() for key, value in self.traces.items()},
            },
            allow_nan=False,
            sort_keys=True,
        )

    @classmethod
    def restore(cls, content: str) -> "PlasticMemory":
        payload = json.loads(content)
        base = payload["base"]
        if payload["version"] != 1:
            raise ValueError("Неизвестная версия checkpoint")
        restored = EncodedMemory.from_state(json.dumps(base), base["representation_id"])
        result = cls(
            base["dimension"],
            base["representation_id"],
            MemoryConfig.model_validate(base["config"]),
            payload["rule"],
        )
        result.__dict__.update(restored.__dict__)
        result.traces = {
            key: np.asarray(value, dtype=float)
            for key, value in payload["traces"].items()
        }
        if set(result.traces) != set(result.pending) or any(
            t.shape != (result.dimension,)
            or not np.isfinite(t).all()
            or not np.isclose(np.linalg.norm(t), 1)
            for t in result.traces.values()
        ):
            raise ValueError("Следы не совпадают с ожидаемыми ответами")
        return result
