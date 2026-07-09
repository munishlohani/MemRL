"""Lightweight types for sleep consolidation.

Two passes, two different decision-makers (spec Project.md "Reflection:
Two-Pass Sleep Consolidation"):
  Pass 1 (structural: spawn/absorb/discard) is algorithmic -- a cosine-
    similarity threshold, no LLM call, no prompt. `SleepConsolidationDecision`
    /`SleepConsolidationResult` carry only `action`/`target_scaffold_id`.
  Pass 2 (content authoring) is the only LLM call, one per *scaffold* with
    changed evidence this sleep event -- see
    `SleepConsolidationService.revise_strategy`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SleepConsolidationAction(str, Enum):
    """Allowed structural outcomes for one tactical cluster (Pass 1)."""

    SPAWN = "spawn"
    ABSORB = "absorb"
    DISCARD = "discard"


@dataclass(frozen=True)
class StrategicScaffoldContext:
    """Context for an existing d=1 scaffold used by Pass 1's cosine check."""

    node_id: str
    summary: str
    embedding: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class SleepConsolidationDecision:
    """Structural decision for one cluster (Pass 1, algorithmic)."""

    action: SleepConsolidationAction
    target_scaffold_id: Optional[str] = None


@dataclass(frozen=True)
class SleepConsolidationResult:
    """Outcome for one clustered sleep-consolidation candidate (Pass 1 only;
    Pass 2's content revision is applied per-scaffold afterward, not per
    cluster -- see `MemoryService.sleep_consolidate`)."""

    cluster_indices: List[int]
    cluster_texts: List[str]
    action: SleepConsolidationAction
    target_scaffold_id: Optional[str] = None
