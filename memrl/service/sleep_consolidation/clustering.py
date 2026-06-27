"""Clustering strategy placeholders for sleep consolidation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence


class ClusteringStrategyBase(ABC):
    """Abstract base class for sleep-consolidation clustering strategies."""

    @abstractmethod
    def cluster(self, embeddings: Sequence[Sequence[float]]) -> List[List[int]]:
        """Group embeddings into index clusters."""


class KMeansClusteringStrategy(ClusteringStrategyBase):
    """Default clustering strategy for Phase 1 sleep consolidation."""

    def cluster(self, embeddings: Sequence[Sequence[float]]) -> List[List[int]]:
        raise NotImplementedError("K-means clustering will be implemented next.")


#Implementation of custom strategies 