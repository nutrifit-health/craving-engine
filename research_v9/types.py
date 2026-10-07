"""Фиксированные параметры и контракты обучения моделей V9."""

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


ModelKind = Literal["mb"]
FoldIndices = tuple[Sequence[int] | np.ndarray, Sequence[int] | np.ndarray]
CalibrationFolds = Sequence[FoldIndices]


@dataclass(frozen=True)
class MBReadoutConfig:
    """Одна заранее заданная логистическая голова без подбора коэффициентов."""

    C: float = 10.0
    max_iter: int = 2000
    tol: float = 1e-6
    fit_intercept: bool = True
    solver: str = "lbfgs"
    class_weight: None = None
    random_state: int = 20260923


@dataclass(frozen=True)
class MBCalibrationConfig:
    """Три независимых fold-модели с sigmoid по исключённым пользователям."""

    mapping_seed: int = 11
    folds: int = 3
    method: Literal["sigmoid"] = "sigmoid"
    ensemble: bool = True
    n_jobs: int = 1
