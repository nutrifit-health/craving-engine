"""Контракты причинной памяти и сохранённого состояния V8.2."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from research_v6.types import ResearchModel
from research_v7.types import MemoryConfig


HeadMode = Literal["off", "on"]
MemoryFormula = Literal["global_tanh_personal_clip_v1"]
MEMORY_FORMULA: MemoryFormula = "global_tanh_personal_clip_v1"


class Snapshot(ResearchModel):
    """Разреженный снимок строго до поступления обратной связи."""

    event_id: str = Field(min_length=1)
    decision_at: AwareDatetime
    indices: tuple[int, ...]
    values: tuple[float, ...]
    p_population: float = Field(gt=0, lt=1)
    p_learning: float = Field(gt=0, lt=1)
    p_candidate: float = Field(gt=0, lt=1)
    p_final: float = Field(gt=0, lt=1)
    gamma: float = Field(ge=0, le=1)
    familiarity: float = Field(ge=0, le=1)
    residual: float = Field(allow_inf_nan=False)
    personal_residual: float = Field(allow_inf_nan=False)
    global_residual: float = Field(allow_inf_nan=False)
    learning_residual: float = Field(allow_inf_nan=False)
    candidate_residual: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_sparse(self) -> Snapshot:
        if len(self.indices) != len(self.values):
            raise ValueError("Индексы и значения ансамбля расходятся")
        if tuple(sorted(set(self.indices))) != self.indices or any(i < 0 for i in self.indices):
            raise ValueError("Индексы ансамбля должны быть уникальными и упорядоченными")
        if any(v <= 0 for v in self.values):
            raise ValueError("Сохранённые активации должны быть положительными")
        if self.residual != self.learning_residual:
            raise ValueError("Residual отчёта должен совпадать с учебной поправкой")
        return self


class Observation(ResearchModel):
    """Глобальному гейту достаточно сохранённых парных ошибок."""

    decision_at: AwareDatetime
    feedback_at: AwareDatetime
    p_population: float = Field(gt=0, lt=1)
    p_candidate: float = Field(gt=0, lt=1)
    outcome: Literal[0, 1]
    confidence: float = Field(gt=0, le=1)


class MemoryState(ResearchModel):
    """Полный checkpoint; привязан к конкретному обученному представлению."""

    version: Literal[2]
    formula: MemoryFormula
    head_mode: HeadMode
    global_head: list[float] | None
    representation_id: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    config: MemoryConfig
    fast: list[float]
    slow: list[float]
    exposure: list[float]
    clock: AwareDatetime | None
    pending: dict[str, Snapshot]
    seen: list[str]
    observations: list[Observation]

    @model_validator(mode="after")
    def validate_head_mode(self) -> MemoryState:
        if (self.head_mode == "off") != (self.global_head is None):
            raise ValueError("Режим головы не соответствует сохранённым весам")
        if self.global_head is not None and len(self.global_head) != self.dimension:
            raise ValueError("Размерность глобальной головы не соответствует памяти")
        return self
