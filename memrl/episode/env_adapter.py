"""Benchmark-neutral environment adapter for episode runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeResetResult:
    """Normalized output from an environment reset."""

    observations: List[str]
    infos: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EpisodeStepResult:
    """Normalized output from one environment step."""

    observations: List[str]
    rewards: List[float]
    dones: List[bool]
    infos: List[Dict[str, Any]] = field(default_factory=list)


class EpisodeEnvAdapter(ABC):
    """
    Normalize benchmark-specific environment APIs for the shared episode runner.

    The runner should only depend on this interface, not on ALFWorld / HLE /
    BigCodeBench / LLB-specific env shapes.
    """

    @abstractmethod
    def reset(self, **kwargs: Any) -> EpisodeResetResult:
        """Start a new episode or batch of episodes."""

    @abstractmethod
    def step(self, actions: List[Any], **kwargs: Any) -> EpisodeStepResult:
        """Advance the environment by one step."""

    @abstractmethod
    def close(self) -> None:
        """Release any environment resources."""

    def episode_id(self, index: int = 0) -> Optional[str]:
        """Return the current episode id when the benchmark exposes one."""
        return None

    def task_type(self, index: int = 0) -> Optional[str]:
        """Return the current task type when the benchmark exposes one."""
        return None

    def is_batch(self) -> bool:
        """Return whether the adapter is managing multiple parallel episodes."""
        return False

    def known_task_types(self) -> List[str]:
        """Return the benchmark's fixed task-type taxonomy, if it has one.

        Used by the runner to pre-seed per-task-type metrics (e.g. success
        rate) at zero so dashboards show every type from the start of a run
        instead of only whatever types early batches happen to sample.
        Empty by default -- benchmarks without a fixed taxonomy (or that
        derive task types dynamically) simply get no pre-seeding.
        """
        return []


__all__ = [
    "EpisodeEnvAdapter",
    "EpisodeResetResult",
    "EpisodeStepResult",
]
