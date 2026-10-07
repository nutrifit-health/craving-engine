"""Контракты экспериментальных абляций; формат V6 не изменяется."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from research_v6.types import MemoryConfig as TemporalConfig
from research_v6.types import ResearchModel


class MemoryConfig(TemporalConfig):
    """Общие параметры для сопоставимых представлений и правил обучения."""

    representation: Literal["raw_kc", "contrast_kc", "contrast_linear", "contrast_scalar"] = "contrast_kc"
    learning: Literal["old", "direct"] = "direct"
    gate: Literal["conditional", "global", "off"] = "global"
    history_share: float = Field(default=0.0, ge=0, le=1)
    max_logit_residual: float = Field(default=0.7, gt=0)
    num_kc: int = Field(default=2048, ge=1)
    fan_in: int = Field(default=6, ge=1)
    winner_fraction: float = Field(default=0.05, gt=0, le=1)


class TransformState(ResearchModel):
    """Параметры, оценённые только на переданных обучающих векторах."""

    version: Literal[1] = 1
    training_rows: int = Field(ge=1)
    mean: tuple[float, ...] = Field(min_length=16, max_length=16)
    std: tuple[float, ...] = Field(min_length=16, max_length=16)
    scale: tuple[float, ...] = Field(min_length=16, max_length=16)
    active_axes: tuple[int, ...]
    std_floor: float = Field(default=0.05, gt=0)
    clip_z: float = Field(default=3.0, gt=0)

    @model_validator(mode="after")
    def validate_statistics(self) -> Self:
        if any(value < 0 for value in self.std):
            raise ValueError("Стандартное отклонение не может быть отрицательным")
        if any(value < 0 or value > 1 for value in self.mean):
            raise ValueError("Средние значения PN должны находиться в [0,1]")
        if any(not math.isclose(scale, max(std, self.std_floor)) for scale, std in zip(self.scale, self.std)):
            raise ValueError("Масштаб должен учитывать фиксированный нижний предел")
        if tuple(sorted(set(self.active_axes))) != self.active_axes or any(axis < 0 or axis >= 16 for axis in self.active_axes):
            raise ValueError("Индексы активных осей должны быть уникальными и упорядоченными")
        return self


class PredictionSnapshot(ResearchModel):
    """Сохранённое решение; residual всегда измеряется в единицах логита."""

    event_id: str = Field(min_length=1)
    decision_at: AwareDatetime
    kc: tuple[float, ...] = Field(min_length=1)
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
            raise ValueError("Активации не могут быть отрицательными")
        norm_squared = sum(value * value for value in self.kc)
        if norm_squared == 0:
            if self.familiarity != 0 or self.gamma != 0 or self.residual != 0:
                raise ValueError("Нулевой контраст не допускает персонализацию")
            if any(value != self.p_population for value in (self.p_learning, self.p_candidate, self.p_final)):
                raise ValueError("Нулевой контраст должен вернуть популяционный прогноз")
        elif not math.isclose(norm_squared, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Ненулевой ансамбль должен иметь единичную L2-норму")
        return self


class GateObservation(ResearchModel):
    """Исход доступен только после сохранённого решения."""

    prediction: PredictionSnapshot
    feedback_at: AwareDatetime
    outcome: Literal[0, 1]
    confidence: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_time(self) -> Self:
        if self.feedback_at <= self.prediction.decision_at:
            raise ValueError("Обратная связь должна следовать за решением")
        return self
