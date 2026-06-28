"""Abstract agent interface for the single-agent MemRL pipeline."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseAgent(ABC):
    """Abstract base class for agents that generate actions for an episode."""

    @abstractmethod
    def reset(self, task_description: str, **kwargs: Any) -> None:
        """Reset agent state for a new episode."""

    @abstractmethod
    def act(
        self,
        observation: str,
        history_messages: List[Dict[str, str]],
        first_step: bool = False,
    ) -> str:
        """Produce the next action from the current observation and history."""

    @abstractmethod
    def get_trajectory(self) -> List[Dict[str, str]]:
        """Return the completed episode trajectory."""
