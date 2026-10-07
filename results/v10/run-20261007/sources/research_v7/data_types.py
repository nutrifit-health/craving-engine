"""Контракты новых выборок V7 с отдельными полями только для оценщика."""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from research_v6.types import ReplayRecord


class V7Record(ReplayRecord):
    """Наблюдаемый feedback и оценочные поля имеют разные роли.

    evaluation_outcome и evaluation_probability доступны исключительно оценщику.
    Модель, online-память, гейт и политика уведомлений их не получают.
    В V9 readout обучается на наблюдаемом training feedback, а не на этих полях.
    """

    split: Literal["training", "calibration", "evaluation"]
    evaluation_probability: float | None = Field(
        default=None,
        ge=0,
        le=1,
        repr=False,
        description="Истинная вероятность генератора, только для диагностик оценщика",
    )


@dataclass(frozen=True)
class SplitSpec:
    """Неизменяемый заранее заданный размер и seed одной выборки."""

    name: Literal["training", "calibration", "evaluation"]
    prefix: str
    seed: int
    users: int
