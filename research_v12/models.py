"""Модели персонализации V12 (PRT-2026-006).

Включает:
1. TunedScalarMemory — настраиваемая скалярная память с честным скользящим окном gate_window_days и JSON-сериализацией.
2. HierarchicalRidgeMemory — компактная контекстная память с эвристическим shrinkage к нулю.
3. SupervisedBottleneckEncoder — контролируемый проектор на остаточный риск (только по наблюдаемым исходам).
4. MixtureOfExpertsModel — смесь популяционных экспертов (обучение строго на наблюдаемых строках).
5. MoEMemory — онлайн-адаптация весов экспертов.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge

from hybrid_engine import safe_logit, safe_sigmoid
from research_v12.state import COMMON_CONFIG, CheckpointMemory


class TunedScalarMemory(CheckpointMemory):
    """Настраиваемая скалярная память V12 с параметризованным темпом, затуханием и защитой."""

    config_fields = COMMON_CONFIG + ("slow_rate",)

    def __init__(
        self,
        fast_rate: float = 0.2,
        slow_rate: float | None = None,
        slow_ratio: float = 0.25,
        l2: float = 0.01,
        fast_half_life_days: float = 14.0,
        slow_half_life_days: float = 90.0,
        exposure_half_life_days: float = 30.0,
        max_logit_residual: float = 0.7,
        gate_window_days: float = 60.0,
        gate_min_effective_samples: float = 12.0,
        gate_min_class_weight: float = 2.0,
        minimum_gain: float = 0.005,
        uncertainty_penalty: float = 1.64,
        gate_temperature: float = 0.008,
        use_gate: bool = True,
    ) -> None:
        self.fast_rate = fast_rate
        self.slow_rate = (
            slow_rate if slow_rate is not None else (fast_rate * slow_ratio)
        )
        self.slow_ratio = slow_ratio
        self.l2 = l2
        self.fast_half_life_days = fast_half_life_days
        self.slow_half_life_days = slow_half_life_days
        self.exposure_half_life_days = exposure_half_life_days
        self.max_logit_residual = max_logit_residual
        self.gate_window_days = gate_window_days
        self.gate_min_effective_samples = gate_min_effective_samples
        self.gate_min_class_weight = gate_min_class_weight
        self.minimum_gain = minimum_gain
        self.uncertainty_penalty = uncertainty_penalty
        self.gate_temperature = gate_temperature
        self.use_gate = use_gate

        self.b_fast = 0.0
        self.b_slow = 0.0
        self.exposure = 0.0
        self.clock: datetime | None = None
        self.pending: dict[str, dict[str, Any]] = {}
        self.observations: list[dict[str, Any]] = []

    def _advance(self, at: datetime) -> None:
        if self.clock is not None:
            elapsed = (at - self.clock).total_seconds() / 86400.0
            if elapsed < 0:
                raise ValueError("Время памяти не может двигаться назад")
            if elapsed > 0:
                dfast = 2.0 ** (-elapsed / self.fast_half_life_days)
                dslow = 2.0 ** (-elapsed / self.slow_half_life_days)
                dexp = 2.0 ** (-elapsed / self.exposure_half_life_days)
                self.b_fast *= dfast
                self.b_slow *= dslow
                self.exposure *= dexp
        self.clock = at
        # Честное удаление наблюдений за пределами скользящего окна gate_window_days
        cutoff_sec = self.gate_window_days * 86400.0
        self.observations = [
            o
            for o in self.observations
            if (at - o["decision_at"]).total_seconds() <= cutoff_sec
        ]

    def _gamma(self) -> float:
        if not self.use_gate:
            return 1.0
        if not self.observations:
            return 0.0
        weights = np.asarray([o["conf"] for o in self.observations], dtype=np.float64)
        y = np.asarray([o["y"] for o in self.observations], dtype=np.float64)
        total = float(weights.sum())
        sum_sq = float(weights @ weights)
        if sum_sq == 0:
            return 0.0
        effective = total * total / sum_sq
        if (
            effective < self.gate_min_effective_samples
            or min(float(weights @ y), float(weights @ (1.0 - y)))
            < self.gate_min_class_weight
        ):
            return 0.0
        gains = np.asarray(
            [
                (o["p0"] - o["y"]) ** 2 - (o["pc"] - o["y"]) ** 2
                for o in self.observations
            ],
            dtype=np.float64,
        )
        mean_gain = float(weights @ gains / total)
        var_gain = float(weights @ ((gains - mean_gain) ** 2) / total)
        uncertainty = math.sqrt(var_gain / max(1.0, effective - 1.0))
        excess = mean_gain - self.minimum_gain - self.uncertainty_penalty * uncertainty
        return (
            0.0 if excess <= 0 else float(-math.expm1(-excess / self.gate_temperature))
        )

    def predict(self, event_id: str, at: datetime, p0: float) -> dict[str, Any]:
        self._advance(at)
        b_res = self.b_fast + self.b_slow
        residual = float(
            np.clip(b_res, -self.max_logit_residual, self.max_logit_residual)
        )
        familiarity = float(-math.expm1(-self.exposure / 3.0))
        gamma = self._gamma()

        base = safe_logit(p0, eps=1e-12)
        candidate = safe_sigmoid(base + familiarity * residual)
        learning = safe_sigmoid(base + residual)
        final = (
            p0
            if gamma == 0 or residual == 0
            else safe_sigmoid(base + gamma * familiarity * residual)
        )

        snap = {
            "event_id": event_id,
            "decision_at": at,
            "p0": p0,
            "p_final": final,
            "p_candidate": candidate,
            "p_learning": learning,
            "gamma": gamma,
            "familiarity": familiarity,
            "residual": residual,
            "b_residual": b_res,
            "w_residual": 0.0,
        }
        self.pending[event_id] = snap
        return snap

    def observe(
        self, event_id: str, at: datetime, outcome: int, confidence: float = 1.0
    ) -> None:
        if event_id not in self.pending:
            return
        snap = self.pending[event_id]
        if at < snap["decision_at"]:
            raise ValueError("Ответ не может предшествовать решению")
        self._advance(at)
        self.pending.pop(event_id)

        age = (at - snap["decision_at"]).total_seconds() / 86400.0
        error = (outcome - snap["p_learning"]) * confidence
        consolidation = self.exposure / (self.exposure + 3.0)

        dfast = 2.0 ** (-age / self.fast_half_life_days)
        dslow = 2.0 ** (-age / self.slow_half_life_days)

        reg_fast = 1.0 - min(1.0, self.fast_rate * self.l2 * confidence)
        reg_slow = 1.0 - min(1.0, self.slow_rate * self.l2 * confidence)

        self.b_fast = self.b_fast * reg_fast + self.fast_rate * dfast * error
        self.b_slow = (
            self.b_slow * reg_slow + self.slow_rate * dslow * consolidation * error
        )
        self.exposure += confidence * 2.0 ** (-age / self.exposure_half_life_days)

        self.observations.append(
            {
                "p0": snap["p0"],
                "pc": snap["p_candidate"],
                "y": outcome,
                "conf": confidence,
                "decision_at": snap["decision_at"],
            }
        )

    def discard(self, event_id: str) -> None:
        self.pending.pop(event_id, None)


# =========================================================================
# Семейство 1: Иерархическая регуляризованная логистическая поправка
# =========================================================================


class HierarchicalRidgeMemory(CheckpointMemory):
    """Семейство 1: регуляризованная поправка с эвристическим притяжением к нулю."""

    config_fields = COMMON_CONFIG + ("dimension", "shrinkage_kappa")

    def __init__(
        self,
        dimension: int = 4,
        fast_rate: float = 0.1,
        slow_ratio: float = 0.25,
        shrinkage_kappa: float = 10.0,
        l2: float = 0.05,
        fast_half_life_days: float = 14.0,
        slow_half_life_days: float = 90.0,
        exposure_half_life_days: float = 30.0,
        max_logit_residual: float = 0.7,
        gate_window_days: float = 60.0,
        gate_min_effective_samples: float = 12.0,
        gate_min_class_weight: float = 2.0,
        minimum_gain: float = 0.005,
        uncertainty_penalty: float = 1.64,
        gate_temperature: float = 0.008,
        use_gate: bool = True,
    ) -> None:
        self.dimension = dimension
        self.fast_rate = fast_rate
        self.slow_rate = fast_rate * slow_ratio
        self.slow_ratio = slow_ratio
        self.shrinkage_kappa = shrinkage_kappa
        self.l2 = l2
        self.fast_half_life_days = fast_half_life_days
        self.slow_half_life_days = slow_half_life_days
        self.exposure_half_life_days = exposure_half_life_days
        self.max_logit_residual = max_logit_residual
        self.gate_window_days = gate_window_days
        self.gate_min_effective_samples = gate_min_effective_samples
        self.gate_min_class_weight = gate_min_class_weight
        self.minimum_gain = minimum_gain
        self.uncertainty_penalty = uncertainty_penalty
        self.gate_temperature = gate_temperature
        self.use_gate = use_gate

        self.b_fast = 0.0
        self.b_slow = 0.0
        self.w_fast = np.zeros(dimension, dtype=np.float64)
        self.w_slow = np.zeros(dimension, dtype=np.float64)
        self.exposure = 0.0
        self.clock: datetime | None = None
        self.pending: dict[str, dict[str, Any]] = {}
        self.observations: list[dict[str, Any]] = []

    def _advance(self, at: datetime) -> None:
        if self.clock is not None:
            elapsed = (at - self.clock).total_seconds() / 86400.0
            if elapsed < 0:
                raise ValueError("Время памяти не может двигаться назад")
            if elapsed > 0:
                dfast = 2.0 ** (-elapsed / self.fast_half_life_days)
                dslow = 2.0 ** (-elapsed / self.slow_half_life_days)
                dexp = 2.0 ** (-elapsed / self.exposure_half_life_days)
                self.b_fast *= dfast
                self.b_slow *= dslow
                self.w_fast *= dfast
                self.w_slow *= dslow
                self.exposure *= dexp
        self.clock = at
        cutoff_sec = self.gate_window_days * 86400.0
        self.observations = [
            o
            for o in self.observations
            if (at - o["decision_at"]).total_seconds() <= cutoff_sec
        ]

    def _gamma(self) -> float:
        if not self.use_gate:
            return 1.0
        if not self.observations:
            return 0.0
        weights = np.asarray([o["conf"] for o in self.observations], dtype=np.float64)
        y = np.asarray([o["y"] for o in self.observations], dtype=np.float64)
        total = float(weights.sum())
        sum_sq = float(weights @ weights)
        if sum_sq == 0:
            return 0.0
        effective = total * total / sum_sq
        if (
            effective < self.gate_min_effective_samples
            or min(float(weights @ y), float(weights @ (1.0 - y)))
            < self.gate_min_class_weight
        ):
            return 0.0
        gains = np.asarray(
            [
                (o["p0"] - o["y"]) ** 2 - (o["pc"] - o["y"]) ** 2
                for o in self.observations
            ],
            dtype=np.float64,
        )
        mean_gain = float(weights @ gains / total)
        var_gain = float(weights @ ((gains - mean_gain) ** 2) / total)
        uncertainty = math.sqrt(var_gain / max(1.0, effective - 1.0))
        excess = mean_gain - self.minimum_gain - self.uncertainty_penalty * uncertainty
        return (
            0.0 if excess <= 0 else float(-math.expm1(-excess / self.gate_temperature))
        )

    def predict(
        self, event_id: str, at: datetime, z: np.ndarray, p0: float
    ) -> dict[str, Any]:
        self._advance(at)
        z_vec = np.asarray(z, dtype=np.float64).copy()
        if z_vec.shape != (self.dimension,) or not np.isfinite(z_vec).all():
            raise ValueError("Некорректная размерность или значения контекста")

        # Эвристический коэффициент shrinkage; не обученный популяционный prior
        shrinkage = self.exposure / (self.exposure + self.shrinkage_kappa)
        b_res = self.b_fast + self.b_slow
        w_res = float(z_vec @ (self.w_fast + self.w_slow)) * shrinkage

        residual = float(
            np.clip(b_res + w_res, -self.max_logit_residual, self.max_logit_residual)
        )
        familiarity = float(-math.expm1(-self.exposure / 3.0))
        gamma = self._gamma()

        base = safe_logit(p0, eps=1e-12)
        candidate = safe_sigmoid(base + familiarity * residual)
        learning = safe_sigmoid(base + residual)
        final = (
            p0
            if gamma == 0 or residual == 0
            else safe_sigmoid(base + gamma * familiarity * residual)
        )

        snap = {
            "event_id": event_id,
            "decision_at": at,
            "z": z_vec,
            "p0": p0,
            "p_final": final,
            "p_candidate": candidate,
            "p_learning": learning,
            "gamma": gamma,
            "familiarity": familiarity,
            "residual": residual,
            "b_residual": b_res,
            "w_residual": w_res,
        }
        self.pending[event_id] = snap
        return snap

    def observe(
        self, event_id: str, at: datetime, outcome: int, confidence: float = 1.0
    ) -> None:
        if event_id not in self.pending:
            return
        snap = self.pending[event_id]
        if at < snap["decision_at"]:
            raise ValueError("Ответ не может предшествовать решению")
        self._advance(at)
        self.pending.pop(event_id)

        age = (at - snap["decision_at"]).total_seconds() / 86400.0
        error = (outcome - snap["p_learning"]) * confidence
        consolidation = self.exposure / (self.exposure + 3.0)

        dfast = 2.0 ** (-age / self.fast_half_life_days)
        dslow = 2.0 ** (-age / self.slow_half_life_days)

        reg_fast = 1.0 - min(1.0, self.fast_rate * self.l2 * confidence)
        reg_slow = 1.0 - min(1.0, self.slow_rate * self.l2 * confidence)

        self.b_fast = self.b_fast * reg_fast + self.fast_rate * dfast * error
        self.b_slow = (
            self.b_slow * reg_slow + self.slow_rate * dslow * consolidation * error
        )

        # Обновление весов w_u с нормировкой на норму z
        z_vec = snap["z"]
        norm_z = float(np.linalg.norm(z_vec))
        scale = 1.0 / max(1.0, norm_z)
        self.w_fast = (
            self.w_fast * reg_fast + self.fast_rate * dfast * error * z_vec * scale
        )
        self.w_slow = (
            self.w_slow * reg_slow
            + self.slow_rate * dslow * consolidation * error * z_vec * scale
        )

        self.exposure += confidence * 2.0 ** (-age / self.exposure_half_life_days)
        self.observations.append(
            {
                "p0": snap["p0"],
                "pc": snap["p_candidate"],
                "y": outcome,
                "conf": confidence,
                "decision_at": snap["decision_at"],
            }
        )

    def discard(self, event_id: str) -> None:
        self.pending.pop(event_id, None)


# =========================================================================
# Семейство 2: Контролируемый остаточный проектор (Supervised Bottleneck)
# =========================================================================


class SupervisedBottleneckEncoder:
    """Семейство 2: Линейный проектор на остаточный риск y - p0 (строго по наблюдаемым строкам)."""

    def __init__(
        self, n_components: int = 4, alpha: float = 10.0, seed: int = 20261201
    ) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.seed = seed
        self.mean_x: np.ndarray | None = None
        self.std_x: np.ndarray | None = None
        self.projection_matrix: np.ndarray | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        p0_train: np.ndarray,
        observed_mask: np.ndarray | None = None,
    ) -> SupervisedBottleneckEncoder:
        if observed_mask is not None:
            obs = np.asarray(observed_mask, dtype=bool)
            X = np.asarray(X_train[obs], dtype=np.float64)
            y = np.asarray(y_train[obs], dtype=np.float64)
            p0 = np.asarray(p0_train[obs], dtype=np.float64)
        else:
            X = np.asarray(X_train, dtype=np.float64)
            y = np.asarray(y_train, dtype=np.float64)
            p0 = np.asarray(p0_train, dtype=np.float64)

        residual = y - p0
        self.mean_x = np.mean(X, axis=0)
        self.std_x = np.where(np.std(X, axis=0) > 1e-4, np.std(X, axis=0), 1.0)
        X_norm = (X - self.mean_x) / self.std_x

        # Вычисляем ковариацию признаков с остаточным риском строго по наблюдаемым строкам
        cov_vector = X_norm.T @ residual / max(1, len(residual))

        rng = np.random.default_rng(self.seed)
        norm_v = float(np.linalg.norm(cov_vector))
        components = [cov_vector / max(1e-6, norm_v)]

        current_res = residual.copy()
        for _ in range(1, self.n_components):
            prev_V = np.column_stack(components)
            Z_prev = X_norm @ prev_V
            ridge = Ridge(alpha=self.alpha).fit(Z_prev, current_res)
            current_res = current_res - ridge.predict(Z_prev)

            new_cov = X_norm.T @ current_res / max(1, len(current_res))
            for comp in components:
                new_cov -= np.dot(new_cov, comp) * comp
            norm_c = float(np.linalg.norm(new_cov))
            if norm_c > 1e-6:
                components.append(new_cov / norm_c)
            else:
                components.append(rng.normal(0, 1.0, size=X.shape[1]))

        self.projection_matrix = np.column_stack(components)[:, : self.n_components]
        return self

    def transform(self, X: np.ndarray, d: int | None = None) -> np.ndarray:
        if self.projection_matrix is None or self.mean_x is None or self.std_x is None:
            raise RuntimeError("Проектор не обучен")
        dim = d or self.n_components
        X_norm = (np.asarray(X, dtype=np.float64) - self.mean_x) / self.std_x
        return X_norm @ self.projection_matrix[:, :dim]


# =========================================================================
# Семейство 3: Смесь популяционных экспертов (Mixture of Experts)
# =========================================================================


class MixtureOfExpertsModel:
    """Семейство 3: Обучение K популяционных моделей (строго на наблюдаемых исходах)."""

    def __init__(self, k_experts: int = 3, seed: int = 20261201) -> None:
        self.k_experts = k_experts
        self.seed = seed
        self.experts: list[HistGradientBoostingClassifier] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        user_ids: list[str],
        observed_mask: np.ndarray | None = None,
    ) -> MixtureOfExpertsModel:
        unique_users = sorted(set(user_ids))
        user_means = []
        for u in unique_users:
            idxs = [i for i, uid in enumerate(user_ids) if uid == u]
            user_means.append(np.mean(X_train[idxs], axis=0))
        user_means_arr = np.asarray(user_means, dtype=np.float64)

        kmeans = KMeans(
            n_clusters=self.k_experts, random_state=self.seed, n_init=10
        ).fit(user_means_arr)
        user_cluster = {u: kmeans.labels_[i] for i, u in enumerate(unique_users)}
        sample_clusters = np.asarray([user_cluster[uid] for uid in user_ids])

        self.experts = []
        for k in range(self.k_experts):
            cluster_mask = sample_clusters == k
            sample_weight = np.where(cluster_mask, 1.0, 0.2)
            if observed_mask is not None:
                obs = np.asarray(observed_mask, dtype=bool)
                X_fit = X_train[obs]
                y_fit = y_train[obs]
                sw_fit = sample_weight[obs]
            else:
                X_fit = X_train
                y_fit = y_train
                sw_fit = sample_weight
            clf = HistGradientBoostingClassifier(
                random_state=self.seed + k, max_iter=80, min_samples_leaf=20
            )
            clf.fit(X_fit, y_fit, sample_weight=sw_fit)
            self.experts.append(clf)
        return self

    def predict_experts(self, X: np.ndarray) -> np.ndarray:
        probs = [clf.predict_proba(X)[:, 1] for clf in self.experts]
        return np.column_stack(probs)


class MoEMemory(CheckpointMemory):
    """Персональная память для Семейства 3: онлайн-обновление вероятностей ответственности alpha."""

    config_fields = ("k_experts", "eta_expert", "half_life_days")

    def __init__(
        self,
        k_experts: int = 3,
        eta_expert: float = 0.5,
        half_life_days: float = 60.0,
    ) -> None:
        self.k_experts = k_experts
        self.eta_expert = eta_expert
        self.half_life_days = half_life_days

        self.log_alpha = np.zeros(k_experts, dtype=np.float64)
        self.clock: datetime | None = None
        self.pending: dict[str, dict[str, Any]] = {}

    def _advance(self, at: datetime) -> None:
        if self.clock is not None:
            elapsed = (at - self.clock).total_seconds() / 86400.0
            if elapsed < 0:
                raise ValueError("Время памяти не может двигаться назад")
            if elapsed > 0:
                decay = 2.0 ** (-elapsed / self.half_life_days)
                self.log_alpha *= decay
        self.clock = at

    def _get_alpha(self) -> np.ndarray:
        max_l = np.max(self.log_alpha)
        e = np.exp(self.log_alpha - max_l)
        return e / np.sum(e)

    def predict(
        self, event_id: str, at: datetime, expert_probs: np.ndarray
    ) -> dict[str, Any]:
        self._advance(at)
        alpha = self._get_alpha()
        p_blend = float(np.dot(alpha, expert_probs))

        snap = {
            "event_id": event_id,
            "decision_at": at,
            "expert_probs": np.asarray(expert_probs, dtype=np.float64).copy(),
            "alpha": alpha,
            "p_final": p_blend,
        }
        self.pending[event_id] = snap
        return snap

    def observe(
        self, event_id: str, at: datetime, outcome: int, confidence: float = 1.0
    ) -> None:
        if event_id not in self.pending:
            return
        snap = self.pending[event_id]
        if at < snap["decision_at"]:
            raise ValueError("Ответ не может предшествовать решению")
        self._advance(at)
        self.pending.pop(event_id)

        probs = snap["expert_probs"]
        lik = np.where(outcome == 1, probs, 1.0 - probs)
        lik = np.clip(lik, 1e-6, 1.0 - 1e-6)

        log_lik = np.log(lik)
        self.log_alpha += self.eta_expert * confidence * log_lik
        # Центрирование для числовой устойчивости
        self.log_alpha -= np.mean(self.log_alpha)

    def discard(self, event_id: str) -> None:
        self.pending.pop(event_id, None)
