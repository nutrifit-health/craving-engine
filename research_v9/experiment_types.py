"""Изолированные задания предобучения и причинного replay V9."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PretrainJob:
    """Fit не получает оценочные исходы и development-признаки."""

    name: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    event_ids: tuple[str, ...]
    folds: list[tuple[np.ndarray, np.ndarray]]
    schema: list[dict]
    graph: np.ndarray
    online_training: np.ndarray
    output: Path


@dataclass(frozen=True)
class ReplayJob:
    """Новый процесс и собственная память для каждого семейства и split."""

    family: str
    split: str
    records_path: Path
    features_path: Path
    population_path: Path | None
    fit_path: Path | None
    input_hashes: dict[str, str]
    output: Path
    training_prevalence: float
    frozen_path: Path | None = None
    frozen_sha256: str | None = None
