"""Write-once skill representation payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SkillRepresentation:
    """Immutable payload for summary content and embedding storage."""

    id: str
    content: str
    embedding: List[float]

    def __post_init__(self) -> None:
        self.embedding = [float(value) for value in self.embedding]
