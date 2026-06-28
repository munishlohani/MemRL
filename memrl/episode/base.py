"""Abstract base class for the episode runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEpisodeRunner(ABC):
    """
    Abstract Base Class for an experiment runner.

    The Runner is responsible for orchestrating the entire interaction between
    the agent, the environment, and any other services (like memory).
    """
    @abstractmethod
    def run(self) -> Any:
        """Run the configured episode pipeline."""


__all__ = ["BaseEpisodeRunner"]
