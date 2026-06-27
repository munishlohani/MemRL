"""Memory service components for the updated MemRL design."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["MemoryService", "ResearchMemoryService"]


def __getattr__(name: str) -> Any:
    if name == "MemoryService":
        return import_module(".memory_service", __name__).MemoryService
    if name == "ResearchMemoryService":
        return import_module(".research_memory_service", __name__).ResearchMemoryService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
