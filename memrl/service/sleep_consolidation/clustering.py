"""Clustering strategy implementations for sleep consolidation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import sqrt
from typing import List, Optional, Sequence

import numpy as np  # type: ignore


from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score


from ..strategies import ClusterStrategy


class ClusteringStrategyBase(ABC):
    """Abstract base class for sleep-consolidation clustering strategies."""

    @abstractmethod
    def cluster(self, embeddings: Sequence[Sequence[float]]) -> List[List[int]]:
        """Group embeddings into index clusters."""


class KMeansClusteringStrategy(ClusteringStrategyBase):
    """Default clustering strategy for Phase 1 sleep consolidation."""

    def __init__(self, n_clusters: Optional[int] = None, random_state: int = 0):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def cluster(self, embeddings: Sequence[Sequence[float]]) -> List[List[int]]:
        vectors = self._coerce_vectors(embeddings)
        if not vectors:
            return []

        vectors = self._normalize_vectors(vectors)
        n_samples = len(vectors)
        if n_samples == 1:
            return [[0]]

        k = self.n_clusters if self.n_clusters is not None else self._default_k(n_samples)
        if self.n_clusters is None:
            k = self._best_k_local(vectors, k)
        if k == 1:
            return [list(range(n_samples))]

        labels = self._run_kmeans(vectors, k)

        clusters: List[List[int]] = [[] for _ in range(k)]
        for idx, label in enumerate(labels):
            clusters[int(label)].append(idx)

        return [cluster for cluster in clusters if cluster]

    @staticmethod
    def _coerce_vectors(embeddings: Sequence[Sequence[float]]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for row in embeddings:
            vectors.append([float(value) for value in row])
        if vectors and any(len(row) != len(vectors[0]) for row in vectors):
            raise ValueError("embeddings must have consistent dimensions")
        return vectors

    @staticmethod
    def _normalize_vectors(vectors: List[List[float]]) -> List[List[float]]:
        normalized: List[List[float]] = []
        for row in vectors:
            norm = sqrt(sum(value * value for value in row))
            denom = norm + 1e-12
            normalized.append([value / denom for value in row])
        return normalized

    @staticmethod
    def _default_k(n_samples: int) -> int:
        """Choose a small, stable default cluster count for Phase 1."""
        if n_samples <= 2:
            return 1
        return max(2, int(sqrt(n_samples)))

    def _run_kmeans(self, vectors: List[List[float]], k: int) -> List[int]:
        try:
            labels = KMeans(
                n_clusters=k,
                n_init="auto",
                random_state=self.random_state,
            ).fit_predict(np.asarray(vectors, dtype=float) if np is not None else vectors)
            return [int(label) for label in labels.tolist()]
        except Exception:
            pass
    
    def _best_k_local(self, vectors: List[List[float]], k_init: int) -> int:
        n_samples = len(vectors)
        if n_samples < 3:
            return max(1, min(k_init, n_samples))

        candidates = {
            max(2, k_init - 1),
            k_init,
            min(n_samples - 1, k_init + 1),
        }
        if not candidates:
            return max(1, min(k_init, n_samples))

        X = np.asarray(vectors, dtype=float) if np is not None else vectors
        best_k = k_init
        best_score = float("inf")

        for k in candidates:
            labels = KMeans(
                n_clusters=k,
                n_init="auto",
                random_state=self.random_state,
            ).fit_predict(X)
            if len(set(int(label) for label in labels.tolist())) < 2:
                continue
            try:
                score = davies_bouldin_score(X, labels)
            except Exception:
                continue

            if score < best_score:
                best_score = score
                best_k = k

        return best_k


class HDBSCANStrategy(ClusteringStrategyBase):
    """Placeholder for the density-based alternative."""

    def cluster(self, embeddings: Sequence[Sequence[float]]) -> List[List[int]]:
        raise NotImplementedError("HDBSCAN clustering will be implemented next.")


def get_clustering_strategy(
    strategy: ClusterStrategy,
    *,
    n_clusters: Optional[int] = None,
    random_state: int = 0,
) -> ClusteringStrategyBase:
    """Factory for sleep-consolidation clustering strategies."""
    if strategy == ClusterStrategy.KMEANS:
        return KMeansClusteringStrategy(n_clusters=n_clusters, random_state=random_state)
    if strategy == ClusterStrategy.HDBSCAN:
        return HDBSCANStrategy()
    raise ValueError(f"Unknown cluster strategy: {strategy}")


__all__ = [
    "ClusteringStrategyBase",
    "KMeansClusteringStrategy",
    "HDBSCANStrategy",
    "get_clustering_strategy",
]
