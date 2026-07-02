"""Shared node model for tactical skills and strategic scaffolds."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Optional

from memrl.utils.q_utils import get_q_salience


@dataclass
class SkillNode:
    """Unified node type used by both tactical and strategic tiers."""

    id: str
    task_type_dominant: str
    t_create: int
    depth: int
    parent_id: Optional[str]
    secondary_parents: list[str] = field(default_factory=list) #not used at this stage ( we have only d=1.2)
    last_accessed_step: int = 0
    Q: Dict[str, float] = field(default_factory=dict)
    n: Dict[str, int] = field(default_factory=dict)
    Q_omega: Dict[str, float] = field(default_factory=dict)
    n_omega: Dict[str, int] = field(default_factory=dict)
    decay_rate: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    evidence_seen: int = 0
    consolidated: bool = False

    def __post_init__(self) -> None:
        if self.depth not in {1, 2}:
            raise ValueError("SkillNode depth must be 1 or 2.")

        if self.depth == 1:
            if self.Q or self.n:
                raise ValueError("Strategic nodes must not populate tactical fields.")
            self.consolidated = False
            self.decay_rate = 0.0
        else:
            if self.Q_omega or self.n_omega:
                raise ValueError("Tactical nodes must not populate strategic fields.")

    @classmethod
    def create_tactical(
        cls,
        *,
        id: str,
        task_type_dominant: str,
        t_create: int,
        parent_id: Optional[str],
        last_accessed_step: int = 0,
        evidence_ids: Optional[list[str]] = None,
    ) -> "SkillNode":
        return cls(
            id=id,
            task_type_dominant=task_type_dominant,
            t_create=t_create,
            depth=2,
            parent_id=parent_id,
            last_accessed_step=last_accessed_step,
            evidence_ids=list(evidence_ids or []),
        )

    @classmethod
    def create_strategic(
        cls,
        *,
        id: str,
        task_type_dominant: str,
        t_create: int,
        parent_id: Optional[str],
        last_accessed_step: int = 0,
        evidence_ids: Optional[list[str]] = None,
    ) -> "SkillNode":
        return cls(
            id=id,
            task_type_dominant=task_type_dominant,
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
        return self.depth == 2

    def q_salience(self, lambda_shrink: float = 10.0) -> float:
        """Return the task-agnostic utility salience used by decay."""
        return get_q_salience(self, lambda_shrink=lambda_shrink)

    def recompute_decay_rate(
        self,
        lambda_base: float,
        epsilon: float,
        lambda_shrink: float,
    ) -> None:
        """Recompute the cached tactical decay rate."""
        if self.depth == 1:
            self.decay_rate = 0.0
            return

        q_bar_w = get_q_salience(self, lambda_shrink=lambda_shrink)
        salience = max(q_bar_w, 0.0)
        self.decay_rate = lambda_base / (salience + epsilon)

    def refresh_task_type_dominant(self) -> None:
        """Recompute task_type_dominant = argmax_k n(t_k) (spec §5.3, §5.4).

        Dynamic, not a formation-time artifact: call whenever this node's
        retrieval counts (`n` for tactical, `n_omega` for strategic) change.
        Ties broken by task type name for determinism. No-op if no task type
        has been observed yet.
        """
        counts = self.n if self.is_tactical else self.n_omega
        if not counts:
            return
        self.task_type_dominant = max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    def add_evidence(self, evidence_id: str, cap: int, rng: Optional[random.Random] = None) -> None:
        """Reservoir-sample `evidence_id` into evidence_ids, capped at `cap` (spec §5.4).

        Classic Algorithm R: every evidence id seen (tracked via
        `evidence_seen`) has an equal cap/evidence_seen chance of surviving
        in the reservoir once it is full.
        """
        picker = rng or random
        self.evidence_seen += 1
        if cap <= 0:
            return
        if len(self.evidence_ids) < cap:
            self.evidence_ids.append(evidence_id)
            return
        j = picker.randint(1, self.evidence_seen)
        if j <= cap:
            self.evidence_ids[picker.randint(0, cap - 1)] = evidence_id
