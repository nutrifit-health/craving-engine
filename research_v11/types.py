"""Типы и структуры данных исследовательской памяти V11."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class MemoryConfigV11(BaseModel):
    """Параметры памяти V11; fast_rate настраивается на tuning, остальные фиксированы."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fast_rate: float = Field(default=0.1, gt=0, le=1.0)
    slow_rate: float = Field(default=0.025, gt=0, le=1.0)
    fast_half_life_days: float = Field(default=14.0, gt=0)
    slow_half_life_days: float = Field(default=90.0, gt=0)
    exposure_half_life_days: float = Field(default=30.0, gt=0)
    l2: float = Field(default=0.01, ge=0)
    max_logit_residual: float = Field(default=0.7, gt=0)

    # Параметры гейта идентичны V8/V9
    gate_window_days: float = Field(default=60.0, gt=0)
    gate_min_effective_samples: float = Field(default=12.0, ge=2.0)
    gate_min_class_weight: float = Field(default=2.0, gt=0)
    minimum_gain: float = Field(default=0.005, ge=0)
    uncertainty_penalty: float = Field(default=1.64, ge=0)
    gate_temperature: float = Field(default=0.008, gt=0)


class SnapshotV11(BaseModel):
    """Снимок решения на момент decision_at для задержанного обновления."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    decision_at: AwareDatetime
    z: tuple[float, ...]
    p_population: float
    p_learning: float
    p_candidate: float
    p_final: float
    gamma: float
    familiarity: float
    residual: float
    b_residual: float
    w_residual: float


class ObservationV11(BaseModel):
    """Наблюдение для скользящего окна гейта."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_at: AwareDatetime
    feedback_at: AwareDatetime
    p_population: float
    p_candidate: float
    outcome: int
    confidence: float


class MemoryStateV11(BaseModel):
    """Полное сериализованное состояние памяти пользователя."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[11] = 11
    dimension: int
    config: MemoryConfigV11
    b_fast: float
    b_slow: float
    w_fast: list[float]
    w_slow: list[float]
    exposure: float
    clock: AwareDatetime | None
    pending: dict[str, SnapshotV11]
    seen: list[str]
    observations: list[ObservationV11]
