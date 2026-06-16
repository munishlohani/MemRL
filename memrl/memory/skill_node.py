"""Shared node model for tactical skills and strategic scaffolds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SkillNode:
    """Unified node type used by both tactical and strategic tiers."""

    id: str
    content: str
    embedding: object
    task_type_primary: str
    t_create: int
    depth: int
    parent_id: Optional[str]
    secondary_parents: List[str] = field(default_factory=list)
    last_accessed_step: int = 0
    Q: Dict[str, float] = field(default_factory=dict)
    n: Dict[str, int] = field(default_factory=dict)
    Q_omega: Dict[str, float] = field(default_factory=dict)
    n_omega: Dict[str, int] = field(default_factory=dict)
    decay_rate: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)
    absorbed_by_sleep: bool = False

    def __post_init__(self) -> None:
        self.embedding = np.asarray(self.embedding, dtype=float)

        if self.depth not in {1, 2, 3}:
            raise ValueError("SkillNode depth must be 1, 2, or 3.")

        if self.depth == 1:
            if self.Q or self.n:
                raise ValueError("Strategic nodes must not populate tactical fields.")
            self.absorbed_by_sleep = False
            self.decay_rate = 0.0
        else:
            if self.Q_omega or self.n_omega:
                raise ValueError("Tactical nodes must not populate strategic fields.")

        if self.depth != 2:
            self.absorbed_by_sleep = False

    @classmethod
    def create_tactical(
        cls,
        *,
        id: str,
        content: str,
        embedding: object,
        task_type_primary: str,
        t_create: int,
        parent_id: Optional[str],
        last_accessed_step: int = 0,
        evidence_ids: Optional[List[str]] = None,
    ) -> "SkillNode":
        return cls(
            id=id,
            content=content,
            embedding=embedding,
            task_type_primary=task_type_primary,
            t_create=t_create,
            depth=3,
            parent_id=parent_id,
            last_accessed_step=last_accessed_step,
            evidence_ids=list(evidence_ids or []),
        )

    @classmethod
    def create_strategic(
        cls,
        *,
        id: str,
        content: str,
        embedding: object,
        task_type_primary: str,
        t_create: int,
        parent_id: Optional[str],
        last_accessed_step: int = 0,
        evidence_ids: Optional[List[str]] = None,
    ) -> "SkillNode":
        return cls(
            id=id,
            content=content,
            embedding=embedding,
            task_type_primary=task_type_primary,
            t_create=t_create,
            depth=1,
            parent_id=parent_id,
            last_accessed_step=last_accessed_step,
            evidence_ids=list(evidence_ids or []),
        )

    @property
    def is_strategic(self) -> bool:
        return self.depth == 1

    @property
    def is_tactical(self) -> bool:
        return not {self.depth==1}

    @property
    def total_accessed(self) -> int:
        """Total tactical retrievals across all task types."""
        return sum(self.n.values())

    def recompute_decay_rate(self, lambda_d: float, epsilon: float) -> None:
        """Recompute the cached tactical decay rate."""
        if self.depth == 1:
            self.decay_rate = 0.0
            return

        q_bar_w = self._weighted_mean_utility(lambda_shrink=10.0)
        self.decay_rate = lambda_d / (q_bar_w + epsilon)

    def _weighted_mean_utility(self, lambda_shrink: float = 10.0) -> float:
        """Confidence-weighted mean over tactical Q values."""
        if not self.Q:
            return 0.0

        weighted_sum = 0.0
        weight_sum = 0.0

        for task_type, q_value in self.Q.items():
            count = self.n.get(task_type, 0)
            weight = count / (count + lambda_shrink)
            weighted_sum += weight * q_value
            weight_sum += weight

        if weight_sum == 0.0:
            return 0.0
        return weighted_sum / weight_sum
