"""Контракты причинного replay и полного состояния исследовательской памяти."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from mushroom_body import NUM_KENYON_CELLS, NUM_PROJECTION_NEURONS


class ResearchModel(BaseModel):
    """Не допускает лишние поля и нечисловые значения вероятностей."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class MemoryConfig(ResearchModel):
    """Заранее фиксируемые параметры кандидата, без автоматического подбора."""

    seed: int = Field(default=42, ge=0)
    beta: float = Field(default=0.35, gt=0, le=1)
    fast_rate: float = Field(default=0.2, gt=0, le=1)
    slow_rate: float = Field(default=0.03, gt=0, le=1)
    fast_half_life_days: float = Field(default=7, gt=0)
    slow_half_life_days: float = Field(default=90, gt=0)
    fast_share: float = Field(default=0.5, ge=0, le=1)
    history_share: float = Field(default=0.25, ge=0, le=1)
    history_half_life_days: float = Field(default=3, gt=0)
    exposure_half_life_days: float = Field(default=90, gt=0)
    l2: float = Field(default=0.001, ge=0)
    max_residual: float = Field(default=2, gt=0)
    gate: Literal["conditional", "global", "off"] = "conditional"
    gate_window_days: float = Field(default=60, gt=0)
    gate_min_effective_samples: float = Field(default=12, ge=2)
    gate_min_class_weight: float = Field(default=2, gt=0)
    minimum_gain: float = Field(default=0.005, ge=0)
    uncertainty_penalty: float = Field(default=1.64, ge=0)
    gate_temperature: float = Field(default=0.008, gt=0)
    similarity_floor: float = Field(default=0.2, ge=0, lt=1)


class ReplayRecord(ResearchModel):
    """Один заранее вычисленный прогноз и отдельно датированный наблюдаемый исход."""

    split: Literal["development"]
    user_id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    decision_at: AwareDatetime
    available_at: AwareDatetime
    p_population: float = Field(gt=0, lt=1)
    pn: list[float] = Field(min_length=NUM_PROJECTION_NEURONS, max_length=NUM_PROJECTION_NEURONS)
    outcome: Literal[0, 1] | None = None
    evaluation_outcome: Literal[0, 1] | None = None
    feedback_at: AwareDatetime | None = None
    confidence: float = Field(default=1, gt=0, le=1)

    @model_validator(mode="after")
    def validate_causality(self) -> Self:
        if self.available_at > self.decision_at:
            raise ValueError("Признаки или популяционный прогноз доступны после решения")
        if any(value < 0 or value > 1 for value in self.pn):
            raise ValueError("Оси PN должны находиться в [0,1]")
        if (self.outcome is None) != (self.feedback_at is None):
            raise ValueError("Наблюдаемый исход требует feedback_at, пропуск — null")
        if self.feedback_at is not None and self.feedback_at <= self.decision_at:
            raise ValueError("Обратная связь должна поступать после прогноза")
        return self


class PredictionSnapshot(ResearchModel):
    """Неизменяемый контекст обучения, сохранённый в момент решения."""

    event_id: str
    decision_at: AwareDatetime
    kc: list[float] = Field(min_length=NUM_KENYON_CELLS, max_length=NUM_KENYON_CELLS)
    p_population: float = Field(gt=0, lt=1)
    p_learning: float = Field(gt=0, lt=1)
    p_candidate: float = Field(gt=0, lt=1)
    p_final: float = Field(gt=0, lt=1)
    familiarity: float = Field(ge=0, le=1)
    gamma: float = Field(ge=0, le=1)
    residual: float

    @model_validator(mode="after")
    def validate_ensemble(self) -> Self:
        if any(value < 0 for value in self.kc):
            raise ValueError("Ансамбль KC не может содержать отрицательные активации")
        if not math.isclose(sum(value * value for value in self.kc), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Ансамбль KC должен быть нормирован по L2")
        return self


class GateObservation(ResearchModel):
    """Парная ошибка до обучения, пригодная только после получения исхода."""

    prediction: PredictionSnapshot
    feedback_at: AwareDatetime
    outcome: Literal[0, 1]
    confidence: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_feedback_time(self) -> Self:
        if self.feedback_at <= self.prediction.decision_at:
            raise ValueError("Обратная связь должна следовать за прогнозом")
        return self


class MemoryState(ResearchModel):
    """Полное состояние отдельной пары пользователь/цель."""

    version: Literal[1] = 1
    config: MemoryConfig
    connectome_sha256: str
    fast: list[float] = Field(min_length=NUM_KENYON_CELLS, max_length=NUM_KENYON_CELLS)
    slow: list[float] = Field(min_length=NUM_KENYON_CELLS, max_length=NUM_KENYON_CELLS)
    exposure: list[float] = Field(min_length=NUM_KENYON_CELLS, max_length=NUM_KENYON_CELLS)
    history: list[float] | None
    clock: AwareDatetime | None
    history_at: AwareDatetime | None
    pending: dict[str, PredictionSnapshot]
    seen: list[str]
    observations: list[GateObservation]


class ReplayPrediction(ResearchModel):
    """Прогноз с отдельными решениями о риске и отправке уведомления."""

    record: ReplayRecord
    probability: float
    candidate_probability: float | None = None
    gamma: float
    familiarity: float
    detected: bool
    notified: bool


class ModelMetrics(ResearchModel):
    """Все доли имеют шкалу [0,1]; неопределяемые метрики остаются null."""

    decisions: int
    observed: int
    scored: int
    label_coverage: float
    events: int
    brier: float | None
    candidate_brier: float | None
    brier_skill_score_retrospective: float | None
    auc: float | None
    recall: float | None
    precision: float | None
    false_positive_rate: float | None
    false_alarms_per_caught: float | None
    notified_recall: float | None
    notified_precision: float | None
    notified_false_alarms_per_caught: float | None
    notifications: int
    mean_gamma: float
    mean_familiarity: float
