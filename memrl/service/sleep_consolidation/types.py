"""Lightweight types for sleep consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class SleepConsolidationAction(str, Enum):
    """Allowed cluster actions produced by the sleep-consolidation LLM."""

    SPAWN = "spawn"
    ABSORB = "absorb"
    DISCARD = "discard"


@dataclass(frozen=True)
class StrategicScaffoldContext:
    """Summary context for an existing d=1 scaffold."""

    node_id: str
    summary: str


@dataclass(frozen=True)
class SleepConsolidationDecision:
    """Structured action returned by the sleep-consolidation LLM."""

    action: SleepConsolidationAction
    summary: Optional[str] = None
    target_scaffold_id: Optional[str] = None


@dataclass(frozen=True)
class SleepConsolidationResult:
    """Outcome for one clustered sleep-consolidation candidate."""

    cluster_indices: List[int]
    cluster_texts: List[str]
    action: SleepConsolidationAction
    summary: Optional[str] = None
    target_scaffold_id: Optional[str] = None
    prompt: str = ""
    raw_response: str = ""
