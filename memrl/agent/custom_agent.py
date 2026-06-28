"""Custom single-pipeline agent implementation."""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Sequence

from memrl.providers.base import BaseLLM

from . import prompts
from .base import BaseAgent

logger = logging.getLogger(__name__)


class CustomAgent(BaseAgent):
    """Concrete BaseAgent implementation for the single-agent pipeline."""

    def __init__(
        self,
        llm_provider: BaseLLM,
        few_shot_examples: Optional[Sequence[Any]] = None,
        *,
        system_prompt: str = prompts.SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm_provider
        self.few_shot_examples = list(few_shot_examples or [])
        self.system_prompt = system_prompt
        self.task_description = ""
        self.task_type = "unknown"
        self.episode_id = "unknown"
        self._trajectory: List[Dict[str, str]] = []

    def reset(self, task_description: str, **kwargs: Any) -> None:
        self.task_description = task_description.strip()
        self.task_type = str(kwargs.get("task_type", "unknown"))
        self.episode_id = str(kwargs.get("episode_id", "unknown"))
        self._trajectory = []
        logger.info(
            "Agent reset for episode=%s task_type=%s",
            self.episode_id,
            self.task_type,
        )

    def act(
        self,
        observation: str,
        history_messages: List[Dict[str, str]],
        first_step: bool = False,
    ) -> str:
        messages = self._build_messages(
            observation=observation,
            history_messages=history_messages,
            first_step=first_step,
        )
        response = self._generate_response(messages)
        action = self._parse_action(response)
        self._trajectory.append(
            {
                "episode_id": self.episode_id,
                "task_type": self.task_type,
                "observation": observation,
                "action": action,
                "response": response or "",
            }
        )
        return action

    def get_trajectory(self) -> List[Dict[str, str]]:
        return list(self._trajectory)

    def _build_messages(
        self,
        *,
        observation: str,
        history_messages: List[Dict[str, str]],
        first_step: bool,
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]

        if self.few_shot_examples:
            rendered_examples = self._render_few_shot_examples(self.few_shot_examples)
            if rendered_examples:
                messages.append(
                    {
                        "role": "system",
                        "content": prompts.FEW_SHOT_PROMPT_SYSTEM.format(
                            few_shot_examples=rendered_examples,
                            retrieved_memories="No archived memories.",
                        ),
                    }
                )

        current_messages = copy.deepcopy(history_messages)
        if not first_step:
            current_messages.append(
                {"role": "user", "content": f"Observation: {observation.strip()}"}
            )

        history_text = self._format_history_messages(current_messages)
        if self.few_shot_examples:
            user_prompt = prompts.FEW_SHOT_PROMPT_USER.format(
                task_description=self.task_description,
                history=history_text,
            )
        else:
            user_prompt = prompts.ZERO_SHOT_PROMPT.format(
                task_description=self.task_description,
                history=history_text,
            )
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @staticmethod
    def _format_history_messages(history_messages: List[Dict[str, str]]) -> str:
        if not history_messages:
            return "You are at the beginning of the task. No steps taken yet."
        lines: List[str] = []
        for message in history_messages:
            role = str(message.get("role", "unknown"))
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "You are at the beginning of the task. No steps taken yet."

    @staticmethod
    def _render_few_shot_examples(examples: Sequence[Any]) -> str:
        rendered: List[str] = []
        for item in examples:
            if isinstance(item, dict):
                if "messages" in item and isinstance(item["messages"], list):
                    rendered.append(CustomAgent._render_message_sequence(item["messages"]))
                    continue
                if "example" in item:
                    example = item["example"]
                    if isinstance(example, list):
                        rendered.append(CustomAgent._render_message_sequence(example))
                        continue
                    rendered.append(str(example))
                    continue
            rendered.append(str(item))
        return "\n\n".join(part for part in rendered if part.strip())

    @staticmethod
    def _render_message_sequence(messages: Sequence[Any]) -> str:
        lines: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "unknown"))
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.llm.generate(messages)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return ""

    @staticmethod
    def _parse_action(response: str) -> str:
        if not response:
            return "look around"
        if "Action:" in response:
            return response.split("Action:", 1)[-1].strip() or "look around"
        return response.strip() or "look around"
