"""Компактные проекторы V11: linear, anatomical KC и rewired KC."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA

from research_v7.connectome import rewire
from research_v8.context_representation import ContextTransform, _graph, _mapping, _raw
from research_v8.replay import graph_matrix


class CompactLinearEncoder:
    """Прямая линейная PCA-проекция наблюдаемых признаков X."""

    def __init__(self, max_components: int = 8, random_state: int = 20261101) -> None:
        self.max_components = max_components
        self.random_state = random_state
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.pca: PCA | None = None
        self.pca_mean: np.ndarray | None = None
        self.pca_std: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> CompactLinearEncoder:
        X = np.asarray(X_train, dtype=np.float64)
        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0)
        self.feature_std = np.where(self.feature_std > 1e-4, self.feature_std, 1.0)

        X_norm = (X - self.feature_mean) / self.feature_std
        self.pca = PCA(n_components=self.max_components, random_state=self.random_state)
        z_raw = self.pca.fit_transform(X_norm)

        self.pca_mean = np.mean(z_raw, axis=0)
        self.pca_std = np.std(z_raw, axis=0)
        self.pca_std = np.where(self.pca_std > 1e-4, self.pca_std, 1.0)
        return self

    def transform(self, X: np.ndarray, d: int) -> np.ndarray:
        if self.pca is None or self.feature_mean is None or self.feature_std is None or self.pca_mean is None or self.pca_std is None:
            raise RuntimeError("Энкодер не обучен")
        if d > self.max_components or d < 1:
            raise ValueError(f"Размерность d должна быть в пределах 1..{self.max_components}")

        X_norm = (np.asarray(X, dtype=np.float64) - self.feature_mean) / self.feature_std
        z = self.pca.transform(X_norm)[:, :d]
        z_std = (z - self.pca_mean[:d]) / self.pca_std[:d]
        return z_std


class CompactKCEncoder:
    """Проекция через Kenyon Cells (2019 нейронов, top-k WTA) с последующим PCA-сжатием."""

    def __init__(
        self,
        graph: np.ndarray,
        family: Literal["anatomical", "rewired"],
        max_components: int = 8,
        k: int = 101,
        seed: int = 20261101,
    ) -> None:
        self.family = family
        self.max_components = max_components
        self.k = k
        self.seed = seed

        _, self.graph_csr = _graph(graph)
        self.context_transform: ContextTransform | None = None
        self.mapping_csr: sparse.csr_matrix | None = None
        self.pca: PCA | None = None
        self.pca_mean: np.ndarray | None = None
        self.pca_std: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, schema: list[dict]) -> CompactKCEncoder:
        X = np.asarray(X_train, dtype=np.float64)
        self.context_transform = ContextTransform.fit(X, schema)
        q_train = self.context_transform.transform(X)

        pn_count = self.graph_csr.shape[1]
        channels = q_train.shape[1]
        mapping = _mapping(pn_count=pn_count, channels=channels, seed=11)
        self.mapping_csr = sparse.csr_matrix(mapping)

        raw_kc = _raw(q_train, self.graph_csr, self.mapping_csr, k=self.k)
        self.pca = PCA(n_components=self.max_components, random_state=self.seed)
        z_raw = self.pca.fit_transform(raw_kc)

        self.pca_mean = np.mean(z_raw, axis=0)
        self.pca_std = np.std(z_raw, axis=0)
        self.pca_std = np.where(self.pca_std > 1e-4, self.pca_std, 1.0)
        return self

    def transform(self, X: np.ndarray, d: int) -> np.ndarray:
        if self.context_transform is None or self.mapping_csr is None or self.pca is None or self.pca_mean is None or self.pca_std is None:
            raise RuntimeError("Энкодер не обучен")
        if d > self.max_components or d < 1:
            raise ValueError(f"Размерность d должна быть в пределах 1..{self.max_components}")

        q = self.context_transform.transform(X)
        raw_kc = _raw(q, self.graph_csr, self.mapping_csr, k=self.k)
        z = self.pca.transform(raw_kc)[:, :d]
        z_std = (z - self.pca_mean[:d]) / self.pca_std[:d]
        return z_std


def build_v11_encoders(
    X_train: np.ndarray,
    schema: list[dict],
    connectome_path: Path,
    rewiring_seed: int = 20260923,
) -> tuple[dict[str, CompactLinearEncoder | CompactKCEncoder], dict[str, object]]:
    """Обучает все 3 компактных семейства строго на обучающей выборке."""
    connectome_path = Path(connectome_path)
    anat_graph, metadata = graph_matrix(connectome_path)
    rewired_graph, rewiring_meta = rewire(anat_graph, seed=rewiring_seed)

    print("Обучение compact_linear...", flush=True)
    linear_enc = CompactLinearEncoder().fit(X_train)

    print("Обучение compact_kc_anatomical...", flush=True)
    anat_enc = CompactKCEncoder(anat_graph, family="anatomical").fit(X_train, schema)

    print("Обучение compact_kc_rewired...", flush=True)
    rewired_enc = CompactKCEncoder(rewired_graph, family="rewired").fit(X_train, schema)

    encoders = {
        "compact_linear": linear_enc,
        "compact_kc_anatomical": anat_enc,
        "compact_kc_rewired": rewired_enc,
    }
    manifest = {
        "max_components": 8,
        "rewiring": rewiring_meta,
        "connectome_metadata": metadata,
    }
    return encoders, manifest
