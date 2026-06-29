"""Abstract agent interface for the single-agent MemRL pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, TypeAlias


@dataclass
class EnvActionDecision:
    """LLM decision to execute an environment action."""

    kind: Literal["env_action"] = "env_action"
    action: str = ""
    thought: str = ""
    raw_response: str = ""

    def as_message(self) -> Dict[str, str]:
        return {
            "role": "assistant",
            "content": self.raw_response.strip()
            or f"Thought: {self.thought}\nAction: {self.action}",
        }

    def __str__(self) -> str:
        return self.action

    def strip(self) -> str:
        return self.action.strip()


@dataclass
class SkillInvocationDecision:
    """LLM decision to invoke a runtime skill."""

    kind: Literal["skill_invocation"] = "skill_invocation"
    skill_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    raw_response: str = ""

    def as_message(self) -> Dict[str, str]:
        return {
            "role": "assistant",
            "content": self.raw_response.strip()
            or f"Thought: {self.thought}\nSkill: {self.skill_name}",
        }

    def __str__(self) -> str:
        return f"Skill: {self.skill_name}"

    def strip(self) -> str:
        return str(self).strip()


AgentDecision: TypeAlias = EnvActionDecision | SkillInvocationDecision


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
    ) -> AgentDecision:
        """Produce either an environment action or a skill invocation."""

    @abstractmethod
    def get_trajectory(self) -> List[Dict[str, str]]:
        """Return the completed episode trajectory."""
