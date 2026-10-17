"""Компактная знаковая контекстная память V11 с точным scalar baseline при d=0."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from hybrid_engine import safe_logit, safe_sigmoid
from research_v11.types import (
    MemoryConfigV11,
    MemoryStateV11,
    ObservationV11,
    SnapshotV11,
)


class CompactMemoryV11:
    """Память пользователя с поддержкой знаковых координат и строгим scalar при d=0."""

    def __init__(self, dimension: int, config: MemoryConfigV11 | None = None) -> None:
        if dimension < 0:
            raise ValueError("Размерность контекста должна быть неотрицательной (0 для scalar)")
        self.dimension = dimension
        self.config = config or MemoryConfigV11()

        # Скалярное смещение (свободный член b_u)
        self.b_fast = 0.0
        self.b_slow = 0.0

        # Контекстные веса w_u
        self.w_fast = np.zeros(dimension, dtype=np.float64)
        self.w_slow = np.zeros(dimension, dtype=np.float64)

        # Скалярная накопленная экспозиция опыта
        self.exposure = 0.0

        self.clock: datetime | None = None
        self.pending: dict[str, SnapshotV11] = {}
        self.seen: set[str] = set()
        self.observations: list[ObservationV11] = []

    def _advance(self, at: datetime) -> None:
        if at.utcoffset() is None:
            raise ValueError("Время должно включать часовой пояс")
        if self.clock is not None:
            elapsed = (at - self.clock).total_seconds() / 86400.0
            if elapsed < 0:
                raise ValueError("События должны поступать строго хронологически")
            decay_fast = 2.0 ** (-elapsed / self.config.fast_half_life_days)
            decay_slow = 2.0 ** (-elapsed / self.config.slow_half_life_days)
            decay_exp = 2.0 ** (-elapsed / self.config.exposure_half_life_days)

            self.b_fast *= decay_fast
            self.b_slow *= decay_slow
            if self.dimension > 0:
                self.w_fast *= decay_fast
                self.w_slow *= decay_slow
            self.exposure *= decay_exp
        self.clock = at
        self.observations = [
            o for o in self.observations
            if (at - o.decision_at).total_seconds() / 86400.0 <= self.config.gate_window_days
        ]

    def _gamma(self) -> float:
        if not self.observations:
            return 0.0
        weights = np.asarray([o.confidence for o in self.observations], dtype=np.float64)
        y = np.asarray([o.outcome for o in self.observations], dtype=np.float64)
        total = float(weights.sum())
        sum_sq = float(weights @ weights)
        if sum_sq == 0:
            return 0.0
        effective = total * total / sum_sq
        if (
            effective < self.config.gate_min_effective_samples
            or min(float(weights @ y), float(weights @ (1.0 - y))) < self.config.gate_min_class_weight
        ):
            return 0.0
        gains = np.asarray(
            [(o.p_population - o.outcome) ** 2 - (o.p_candidate - o.outcome) ** 2 for o in self.observations],
            dtype=np.float64,
        )
        mean = float(weights @ gains / total)
        variance = float(weights @ ((gains - mean) ** 2) / total)
        uncertainty = math.sqrt(variance / max(1.0, effective - 1.0))
        excess = mean - self.config.minimum_gain - self.config.uncertainty_penalty * uncertainty
        return 0.0 if excess <= 0 else float(-math.expm1(-excess / self.config.gate_temperature))

    def predict(
        self,
        event_id: str,
        at: datetime,
        z: np.ndarray | None,
        p_population: float,
    ) -> SnapshotV11:
        """Принимает доступный контекст z и популяционный прогноз p_0."""
        if not event_id or event_id in self.seen:
            raise ValueError("Пустой либо повторный event_id")
        if not 0 < p_population < 1:
            raise ValueError("Некорректная популяционная вероятность")

        if self.dimension == 0:
            if z is not None and len(z) > 0 and not (len(z) == 1 and z[0] == 1.0):
                raise ValueError("При dimension=0 контекст z должен быть None, пустым или [1.0]")
            z_vec = np.zeros(0, dtype=np.float64)
        else:
            if z is None:
                raise ValueError(f"Ожидался контекстный вектор размерности {self.dimension}")
            z_vec = np.asarray(z, dtype=np.float64)
            if z_vec.shape != (self.dimension,) or not np.isfinite(z_vec).all():
                raise ValueError(f"Контекст z должен иметь форму ({self.dimension},) и конечные значения")

        self._advance(at)

        cap = self.config.max_logit_residual
        b_res = self.b_fast + self.b_slow
        w_res = float(z_vec @ (self.w_fast + self.w_slow)) if self.dimension > 0 else 0.0
        personal_residual = float(np.clip(b_res + w_res, -cap, cap))

        familiarity = float(-math.expm1(-self.exposure / 3.0))
        gamma = self._gamma()
        base = safe_logit(p_population, eps=1e-12)

        candidate_residual = familiarity * personal_residual
        learning = safe_sigmoid(base + personal_residual)
        candidate = safe_sigmoid(base + candidate_residual)
        final = (
            p_population
            if gamma == 0 or personal_residual == 0
            else safe_sigmoid(base + gamma * familiarity * personal_residual)
        )

        snap = SnapshotV11(
            event_id=event_id,
            decision_at=at,
            z=tuple(float(v) for v in z_vec),
            p_population=p_population,
            p_learning=learning,
            p_candidate=candidate,
            p_final=final,
            gamma=gamma,
            familiarity=familiarity,
            residual=personal_residual,
            b_residual=b_res,
            w_residual=w_res,
        )
        self.pending[event_id] = snap
        self.seen.add(event_id)
        return snap

    def observe(self, event_id: str, at: datetime, outcome: int, confidence: float = 1.0) -> None:
        """Обновляет память по сохраненному снимку решения."""
        if event_id not in self.pending:
            raise ValueError("Неизвестный либо повторный feedback")
        snap = self.pending[event_id]
        if outcome not in (0, 1) or not (0 < confidence <= 1) or at.utcoffset() is None or at <= snap.decision_at:
            raise ValueError("Некорректный feedback")

        self._advance(at)

        consolidation = self.exposure / (self.exposure + 3.0)
        age = (at - snap.decision_at).total_seconds() / 86400.0
        error = (outcome - snap.p_learning) * confidence

        # Регуляризация (L2 dampening) идентично V8
        reg_factor_fast = 1.0 - min(1.0, self.config.fast_rate * self.config.l2 * confidence)
        reg_factor_slow = 1.0 - min(1.0, self.config.slow_rate * self.config.l2 * confidence)

        self.b_fast *= reg_factor_fast
        self.b_slow *= reg_factor_slow

        d_fast = 2.0 ** (-age / self.config.fast_half_life_days)
        d_slow = 2.0 ** (-age / self.config.slow_half_life_days)

        self.b_fast += self.config.fast_rate * d_fast * error
        self.b_slow += self.config.slow_rate * d_slow * consolidation * error

        if self.dimension > 0 and len(snap.z) > 0:
            z_vec = np.asarray(snap.z, dtype=np.float64)
            self.w_fast *= reg_factor_fast
            self.w_slow *= reg_factor_slow
            self.w_fast += self.config.fast_rate * d_fast * error * z_vec
            self.w_slow += self.config.slow_rate * d_slow * consolidation * error * z_vec

        self.exposure += confidence * 2.0 ** (-age / self.config.exposure_half_life_days)
        self.observations.append(
            ObservationV11(
                decision_at=snap.decision_at,
                feedback_at=at,
                p_population=snap.p_population,
                p_candidate=snap.p_candidate,
                outcome=outcome,
                confidence=confidence,
            )
        )
        del self.pending[event_id]

    def discard(self, event_id: str) -> None:
        """Отсутствие ответа удаляет снимок без создания негативной метки."""
        self.pending.pop(event_id, None)

    def export_state(self) -> str:
        """Полная сериализация состояния в JSON."""
        return MemoryStateV11(
            version=11,
            dimension=self.dimension,
            config=self.config,
            b_fast=float(self.b_fast),
            b_slow=float(self.b_slow),
            w_fast=[float(v) for v in self.w_fast],
            w_slow=[float(v) for v in self.w_slow],
            exposure=float(self.exposure),
            clock=self.clock,
            pending=self.pending,
            seen=sorted(self.seen),
            observations=self.observations,
        ).model_dump_json()

    @classmethod
    def from_state(cls, payload: str) -> CompactMemoryV11:
        state = MemoryStateV11.model_validate_json(payload)
        instance = cls(state.dimension, state.config)
        instance.b_fast = state.b_fast
        instance.b_slow = state.b_slow
        instance.w_fast = np.asarray(state.w_fast, dtype=np.float64)
        instance.w_slow = np.asarray(state.w_slow, dtype=np.float64)
        instance.exposure = state.exposure
        instance.clock = state.clock
        instance.pending = dict(state.pending)
        instance.seen = set(state.seen)
        instance.observations = list(state.observations)
        return instance

    def state_size_bytes(self) -> int:
        """Измеряет размер сериализованного состояния в байтах UTF-8."""
        return len(self.export_state().encode("utf-8"))
