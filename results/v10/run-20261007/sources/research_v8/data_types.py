"""Контракты замороженного корпуса и наблюдаемого обучения V8.1."""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from research_v7.data_types import V7Record


class V8Record(V7Record):
    """Старый evaluation используется как development, без заявления holdout.

    Сохранённый популяционный прогноз V7 исключается при загрузке. Новый прогноз
    появляется только после обучения на наблюдаемом training feedback.
    evaluation_outcome и evaluation_probability остаются полями оценщика.
    """

    split: Literal["training", "calibration", "development"]
    p_population: float | None = Field(default=None, gt=0, lt=1)


@dataclass(frozen=True)
class CorpusSplit:
    """Заранее зафиксированная связь исходного файла и роли в V8.1."""

    source: str
    role: str
    prefix: str
    users: int
    seed: int


@dataclass(frozen=True)
class PopulationConfig:
    """Единая конфигурация без выбора параметров по calibration/development."""

    max_iter: int = 150
    max_leaf_nodes: int = 15
    learning_rate: float = 0.05
    l2_regularization: float = 1.0
    random_state: int = 20260923

