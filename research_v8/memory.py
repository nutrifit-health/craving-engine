"""Общая direct-пластичность V8.2 для заранее вычисленных представлений.

Необязательная глобальная голова заморожена отдельно от персональной памяти.
Персональные синапсы всегда начинаются с нуля; head_off сохраняет математику V8.1.
"""

from __future__ import annotations

from datetime import datetime
import math

import numpy as np

from hybrid_engine import safe_logit, safe_sigmoid
from research_v7.types import MemoryConfig
from research_v8.types import MEMORY_FORMULA, HeadMode, MemoryState, Observation, Snapshot


class EncodedMemory:
    """Память одного пользователя; глобальный гейт не зависит от размерности KC."""

    def __init__(
        self, dimension: int, representation_id: str, config: MemoryConfig | None = None,
        *, global_head: np.ndarray | None = None,
    ) -> None:
        if dimension < 1 or not representation_id:
            raise ValueError("Требуются размерность и идентификатор представления")
        self.config = config or MemoryConfig()
        if self.config.gate != "global" or self.config.learning != "direct" or self.config.history_share != 0:
            raise ValueError("V8.2 фиксирует direct, global gate и отсутствие истории")
        self.dimension, self.representation_id = dimension, representation_id
        self._global_head = None
        if global_head is not None:
            head = np.array(global_head, dtype=np.float64, copy=True)
            if head.shape != (dimension,) or not np.isfinite(head).all():
                raise ValueError("Глобальная голова должна быть конечным вектором размерности памяти")
            # Неизменяемый буфер исключает запись и обратное включение writeable.
            self._global_head = np.frombuffer(head.tobytes(), dtype=np.float64)
        self.fast = np.zeros(dimension)
        self.slow = np.zeros(dimension)
        self.exposure = np.zeros(dimension)
        self.clock: datetime | None = None
        self.pending: dict[str, Snapshot] = {}
        self.seen: set[str] = set()
        self.observations: list[Observation] = []

    @property
    def global_head(self) -> np.ndarray | None:
        """Возвращает замороженные веса; None означает отсутствие глобальной головы."""
        return self._global_head

    @property
    def head_mode(self) -> HeadMode:
        """Нулевая переданная голова остаётся явным режимом on."""
        return "off" if self._global_head is None else "on"

    def _advance(self, at: datetime) -> None:
        if at.utcoffset() is None:
            raise ValueError("Время должно включать часовой пояс")
        if self.clock is not None:
            elapsed = (at - self.clock).total_seconds() / 86400
            if elapsed < 0:
                raise ValueError("События должны поступать хронологически")
            self.fast *= 2 ** (-elapsed / self.config.fast_half_life_days)
            self.slow *= 2 ** (-elapsed / self.config.slow_half_life_days)
            self.exposure *= 2 ** (-elapsed / self.config.exposure_half_life_days)
        self.clock = at
        self.observations = [o for o in self.observations if (at - o.decision_at).total_seconds() / 86400 <= self.config.gate_window_days]

    def _gamma(self) -> float:
        if not self.observations:
            return 0.0
        weights = np.asarray([o.confidence for o in self.observations])
        y = np.asarray([o.outcome for o in self.observations])
        total = float(weights.sum())
        effective = total * total / float(weights @ weights)
        if effective < self.config.gate_min_effective_samples or min(float(weights @ y), float(weights @ (1 - y))) < self.config.gate_min_class_weight:
            return 0.0
        gains = np.asarray([(o.p_population - o.outcome) ** 2 - (o.p_candidate - o.outcome) ** 2 for o in self.observations])
        mean = float(weights @ gains / total)
        variance = float(weights @ ((gains - mean) ** 2) / total)
        uncertainty = math.sqrt(variance / max(1.0, effective - 1))
        excess = mean - self.config.minimum_gain - self.config.uncertainty_penalty * uncertainty
        return 0.0 if excess <= 0 else float(-math.expm1(-excess / self.config.gate_temperature))

    def predict(self, event_id: str, at: datetime, encoded: np.ndarray, p_population: float) -> Snapshot:
        """Принимает только доступные признаки и популяционный прогноз."""
        h = np.asarray(encoded, dtype=float)
        if not event_id or event_id in self.seen:
            raise ValueError("Пустой либо повторный event_id")
        if h.shape != (self.dimension,) or not np.isfinite(h).all() or np.any(h < 0):
            raise ValueError("Некорректное представление")
        norm = float(np.linalg.norm(h))
        if norm and not math.isclose(norm, 1, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Ненулевой ансамбль должен иметь единичную норму")
        if not 0 < p_population < 1:
            raise ValueError("Некорректная популяционная вероятность")
        self._advance(at)
        indices = np.flatnonzero(h)
        personal_residual = global_residual = learning_residual = candidate_residual = 0.0
        familiarity, gamma = 0.0, 0.0
        learning = candidate = final = p_population
        if len(indices):
            cap = self.config.max_logit_residual
            personal_residual = float(np.clip(h @ (self.fast + self.slow), -cap, cap))
            familiarity = float(-math.expm1(-float(np.mean(self.exposure[indices])) / 3))
            gamma = self._gamma()
            base = safe_logit(p_population, eps=1e-12)
            if self._global_head is None:
                learning_residual = personal_residual
                candidate_residual = familiarity * personal_residual
                learning = safe_sigmoid(base + learning_residual)
                candidate = safe_sigmoid(base + candidate_residual)
                # Порядок умножений сохраняет точное численное поведение V8.1.
                final = p_population if gamma == 0 or personal_residual == 0 else safe_sigmoid(base + gamma * familiarity * personal_residual)
            else:
                global_residual = cap * math.tanh(float(h @ self._global_head) / cap)
                learning_residual = float(np.clip(global_residual + personal_residual, -cap, cap))
                candidate_residual = float(np.clip(global_residual + familiarity * personal_residual, -cap, cap))
                learning = safe_sigmoid(base + learning_residual)
                candidate = safe_sigmoid(base + candidate_residual)
                final = p_population if gamma == 0 or candidate_residual == 0 else safe_sigmoid(base + gamma * candidate_residual)
        snap = Snapshot(
            event_id=event_id, decision_at=at, indices=tuple(int(i) for i in indices), values=tuple(h[indices]),
            p_population=p_population, p_learning=learning, p_candidate=candidate, p_final=final,
            gamma=gamma, familiarity=familiarity, residual=learning_residual,
            personal_residual=personal_residual, global_residual=global_residual,
            learning_residual=learning_residual, candidate_residual=candidate_residual,
        )
        self.pending[event_id] = snap
        self.seen.add(event_id)
        return snap

    def observe(self, event_id: str, at: datetime, outcome: int, confidence: float = 1.0) -> None:
        """Обновляет исходный ансамбль по сохранённой ошибке всего гибрида."""
        if event_id not in self.pending:
            raise ValueError("Неизвестный либо повторный feedback")
        snap = self.pending[event_id]
        if outcome not in (0, 1) or not 0 < confidence <= 1 or at.utcoffset() is None or at <= snap.decision_at:
            raise ValueError("Некорректный feedback")
        self._advance(at)
        indices = np.asarray(snap.indices, dtype=int)
        if len(indices):
            values = np.asarray(snap.values)
            experience = float(np.mean(self.exposure[indices]))
            consolidation = experience / (experience + 3)
            age = (at - snap.decision_at).total_seconds() / 86400
            error = (outcome - snap.p_learning) * confidence
            self.fast *= 1 - min(1.0, self.config.fast_rate * self.config.l2 * confidence)
            self.slow *= 1 - min(1.0, self.config.slow_rate * self.config.l2 * confidence)
            self.fast[indices] += self.config.fast_rate * 2 ** (-age / self.config.fast_half_life_days) * error * values
            self.slow[indices] += self.config.slow_rate * 2 ** (-age / self.config.slow_half_life_days) * consolidation * error * values
            self.exposure[indices] += confidence * 2 ** (-age / self.config.exposure_half_life_days)
            self.observations.append(Observation(decision_at=snap.decision_at, feedback_at=at, p_population=snap.p_population, p_candidate=snap.p_candidate, outcome=outcome, confidence=confidence))
        del self.pending[event_id]

    def discard(self, event_id: str) -> None:
        """Отсутствие ответа не создаёт отрицательной метки."""
        self.pending.pop(event_id, None)

    def export_state(self) -> str:
        """Сохраняет все части состояния, включая ожидающие ответы и dedup."""
        return MemoryState(
            version=2, formula=MEMORY_FORMULA, head_mode=self.head_mode,
            global_head=None if self._global_head is None else self._global_head.tolist(),
            representation_id=self.representation_id, dimension=self.dimension, config=self.config,
            fast=self.fast.tolist(), slow=self.slow.tolist(), exposure=self.exposure.tolist(),
            clock=self.clock, pending=self.pending, seen=sorted(self.seen), observations=self.observations,
        ).model_dump_json()

    @classmethod
    def from_state(
        cls, payload: str, expected_representation_id: str,
        *, expected_global_head: np.ndarray | None = None,
    ) -> EncodedMemory:
        """Сверяет checkpoint с внешней головой; None требует режим head_off."""
        state = MemoryState.model_validate_json(payload)
        if state.representation_id != expected_representation_id:
            raise ValueError("Checkpoint принадлежит другому представлению")
        expected_mode = "off" if expected_global_head is None else "on"
        if state.head_mode != expected_mode:
            raise ValueError("Checkpoint принадлежит другому режиму глобальной головы")
        if expected_global_head is not None and not np.array_equal(
            np.asarray(state.global_head, dtype=float), np.asarray(expected_global_head, dtype=float),
        ):
            raise ValueError("Глобальная голова checkpoint не совпадает с ожидаемыми весами")
        result = cls(
            state.dimension, state.representation_id, state.config,
            global_head=expected_global_head,
        )
        arrays = [np.asarray(a, dtype=float) for a in (state.fast, state.slow, state.exposure)]
        if any(a.shape != (state.dimension,) or not np.isfinite(a).all() for a in arrays) or np.any(arrays[2] < 0):
            raise ValueError("Некорректные веса checkpoint")
        if len(set(state.seen)) != len(state.seen) or not set(state.pending) <= set(state.seen):
            raise ValueError("Некорректное состояние dedup")
        for event_id, snap in state.pending.items():
            if event_id != snap.event_id or any(i >= state.dimension for i in snap.indices):
                raise ValueError("Некорректный ожидающий ансамбль")
            if snap.values and not math.isclose(sum(v*v for v in snap.values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("Ненормированный ожидающий ансамбль")
            if state.clock is None or snap.decision_at > state.clock:
                raise ValueError("Ожидающий ансамбль создан в будущем")
        if any(state.clock is None or o.feedback_at > state.clock or o.feedback_at <= o.decision_at for o in state.observations):
            raise ValueError("Некорректное время наблюдений гейта")
        result.fast, result.slow, result.exposure = arrays
        result.clock, result.pending = state.clock, state.pending
        result.seen, result.observations = set(state.seen), state.observations
        return result
