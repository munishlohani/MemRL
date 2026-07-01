"""Episodic memory bank: raw experience store linked from SkillNode.evidence_ids.

Nodes never surface raw experience directly at retrieval time -- only their
LLM-generated `skill_representation.content` (spec §1, §5.3, §5.4). This
bank keeps the underlying per-step trace available for inspection and for
future causal credit-assignment work (spec §11), addressed by the same ids
stored in `SkillNode.evidence_ids`. `SkillNode.add_evidence` reservoir-caps
which ids a node keeps; this bank is the store those ids point into.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EpisodicRecord:
    """One raw experience: a single step's observation/action/outcome."""

    id: str
    task_type: str
    task_description: str
    episode_id: str
    step_index: int
    observation: str
    action: str
    reward: float
    history: str
    retrieved_memories: str
    source_memory_id: Optional[str] = None


@dataclass
class EpisodicMemoryBank:
    """In-memory store of `EpisodicRecord`, keyed by id.

    Synced to SQLite by `MemoryService` (`episodic_memory` table), mirroring
    how `SkillGraph` is the in-memory structure that `MemoryService` persists
    for `skill_graph_state`.
    """

    records: Dict[str, EpisodicRecord] = field(default_factory=dict)

    def add(self, record: EpisodicRecord) -> None:
        self.records[record.id] = record

    def has(self, record_id: str) -> bool:
        return record_id in self.records

    def get(self, record_id: str) -> Optional[EpisodicRecord]:
        return self.records.get(record_id)

    def get_many(self, record_ids: List[str]) -> List[EpisodicRecord]:
        return [self.records[rid] for rid in record_ids if rid in self.records]

    def __len__(self) -> int:
        return len(self.records)


__all__ = ["EpisodicRecord", "EpisodicMemoryBank"]
