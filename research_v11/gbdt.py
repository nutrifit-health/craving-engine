"""Обучение базовой популяционной модели GBDT (p_0) строго на training."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class PopulationGBDTTrainer:
    """Обучает воспроизводимый HistGradientBoostingClassifier на наблюдаемых training-исходах."""

    def __init__(self, random_state: int = 20261101, max_iter: int = 100, min_samples_leaf: int = 20) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.model = HistGradientBoostingClassifier(
            random_state=self.random_state,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, observed_mask: np.ndarray) -> PopulationGBDTTrainer:
        """Обучает модель только на тех строках training, где исход наблюдаем (reported)."""
        X_obs = X_train[observed_mask]
        y_obs = y_train[observed_mask]
        if len(X_obs) == 0 or len(set(y_obs)) < 2:
            raise ValueError("Недостаточно наблюдаемых обучающих строк обоих классов")
        self.model.fit(X_obs, y_obs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает популяционную вероятность p_0(x) для каждого наблюдения."""
        p = self.model.predict_proba(X)[:, 1]
        return np.clip(p, 1e-4, 1.0 - 1e-4)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self.model, stream)

    @classmethod
    def load(cls, path: Path) -> PopulationGBDTTrainer:
        instance = cls()
        with Path(path).open("rb") as stream:
            instance.model = pickle.load(stream)
        return instance
