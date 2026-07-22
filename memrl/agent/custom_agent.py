"""Custom single-pipeline agent implementation."""

from __future__ import annotations

import ast
import logging
from typing import Any, Dict, List, Optional, Sequence

from memrl.providers.base import BaseLLM
from memrl.skills.memory_retrieval import MemoryRetrievalResult, MemoryRetrievalSkill

from . import prompts
from .base import AgentDecision, BaseAgent, EnvActionDecision, SkillInvocationDecision

logger = logging.getLogger(__name__)


class CustomAgent(BaseAgent):
    """Concrete BaseAgent implementation for the single-agent pipeline."""

    def __init__(
        self,
        llm_provider: BaseLLM,
        few_shot_examples: Optional[Sequence[Any]] = None,
        *,
        system_prompt: str = prompts.SYSTEM_PROMPT,
        memory_retrieval_skill: Optional[MemoryRetrievalSkill] = None,
    ) -> None:
        # few_shot_examples is accepted-but-discarded: kept in the signature
        # so existing callers (run_alfworld.py/run_bcb.py/run_alfworld_ray.py)
        # don't break, but the mechanism was removed -- it only ever
        # rendered via the now-removed few-shot system message.
        del few_shot_examples
        self.llm = llm_provider
        self.system_prompt = system_prompt
        self.memory_retrieval_skill = memory_retrieval_skill
        self.task_description = ""
        self.task_type = "unknown"
        self.episode_id = "unknown"
        self._trajectory: List[Dict[str, str]] = []
        self.last_memory_retrieval: Optional[Dict[str, Any]] = None
        self.last_decision: Optional[AgentDecision] = None

    def reset(self, task_description: str, **kwargs: Any) -> None:
        self.task_description = task_description.strip()
        self.task_type = str(kwargs.get("task_type", "unknown"))
        self.episode_id = str(kwargs.get("episode_id", "unknown"))
        self._trajectory = []
        self.last_memory_retrieval = None
        self.last_decision = None
        logger.info(
            "Agent reset for episode=%s task_type=%s",
            self.episode_id,
            self.task_type,
        )

    def act(
        self,
        observation: str,
        history_messages: List[Dict[str, Any]],
        first_step: bool = False,
        **kwargs: Any,
    ) -> AgentDecision:
        messages = self._build_messages(
            observation=observation,
            history_messages=history_messages,
            first_step=first_step,
            admissible_commands=kwargs.get("admissible_commands") or [],
        )
        response = self._generate_response(messages)
        decision = self._parse_decision(response)
        self.last_decision = decision
        self._trajectory.append(
            {
                "episode_id": self.episode_id,
                "task_type": self.task_type,
                "observation": observation,
                "action": getattr(decision, "action", str(decision)),
                "decision_kind": getattr(decision, "kind", "unknown"),
                "skill_name": getattr(decision, "skill_name", ""),
                "response": response or "",
            }
        )
        return decision

    def record_memory_retrieval(self, retrieval_result: MemoryRetrievalResult) -> None:
        """Record the most recent memory retrieval result for logging."""
        self.last_memory_retrieval = retrieval_result.to_dict()

    def get_trajectory(self) -> List[Dict[str, str]]:
        return list(self._trajectory)

    def _build_messages(
        self,
        *,
        observation: str,
        history_messages: List[Dict[str, Any]],
        first_step: bool,
        admissible_commands: Sequence[str] = (),
    ) -> List[Dict[str, str]]:
        admissible_text = (
            "\n".join(f"- {command}" for command in admissible_commands)
            if admissible_commands
            else "(none available)"
        )

        skill_contract = self._render_memory_retrieval_contract()
        system_content = (
            self.system_prompt.format(skill_contract=skill_contract)
            if "{skill_contract}" in self.system_prompt
            else self.system_prompt
        )
        messages = [{"role": "system", "content": system_content}]

        history_text = self._format_history_messages(history_messages)
        if skill_contract:
            user_prompt = prompts.SKILL_AWARE_PROMPT.format(
                task_description=self.task_description,
                observation=observation,
                history=history_text,
                admissible=admissible_text,
            )
        else:
            user_prompt = prompts.ZERO_SHOT_PROMPT.format(
                task_description=self.task_description,
                observation=observation,
                history=history_text,
                admissible=admissible_text,
            )
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _render_memory_retrieval_contract(self) -> str:
        skill = self.memory_retrieval_skill
        if skill is None:
            return ""

        contract = getattr(skill, "prompt_contract", None)
        if callable(contract):
            try:
                contract = contract()
            except Exception:
                logger.debug("Failed to render memory retrieval contract", exc_info=True)
                contract = ""
        if contract is None:
            contract = getattr(skill, "skill_contract", "")
        return str(contract or "").strip()

    _MAX_RECENT_HISTORY_MESSAGES = 20

    @staticmethod
    def _format_history_messages(history_messages: List[Dict[str, Any]]) -> str:
        if not history_messages:
            return "You are at the beginning of the task. No steps taken yet."

        max_recent = CustomAgent._MAX_RECENT_HISTORY_MESSAGES
        if len(history_messages) > max_recent:
            # Was previously fully unbounded -- every memory_retrieval tool
            # result (often sizable) accumulated in this list forever, so a
            # long episode's context grew without limit. Pin message[0] (the
            # room's initial description, appended once by EpisodeRunner.run)
            # regardless of length, plus the most recent max_recent-1
            # messages, so the room layout can't silently fall out of
            # context even when the tail is trimmed.
            selected = [history_messages[0]] + history_messages[-(max_recent - 1):]
        else:
            selected = history_messages

        lines: List[str] = []
        for message in selected:
            role = str(message.get("role", "unknown"))
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            name = str(message.get("name", "")).strip()
            if role == "tool" and name:
                lines.append(f"{role}[{name}]: {content}")
            else:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "You are at the beginning of the task. No steps taken yet."

    def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.llm.generate(messages)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return ""

    @staticmethod
    def _parse_decision(response: str) -> AgentDecision:
        text = (response or "").strip()
        if not text:
            return EnvActionDecision(action="look around", raw_response="")

        skill_directive = CustomAgent._extract_directive(text, "Skill:")
        action_directive = CustomAgent._extract_directive(text, "Action:")
        thought = CustomAgent._extract_directive(text, "Thought:")

        skill_index = text.find("Skill:") if "Skill:" in text else -1
        action_index = text.find("Action:") if "Action:" in text else -1

        if skill_directive and (action_index == -1 or (skill_index != -1 and skill_index < action_index)):
            skill_name, arguments = CustomAgent._parse_skill_directive(skill_directive)
            return SkillInvocationDecision(
                skill_name=skill_name,
                arguments=arguments,
                thought=thought,
                raw_response=text,
            )

        if action_directive:
            return EnvActionDecision(
                action=action_directive,
                thought=thought,
                raw_response=text,
            )

        return EnvActionDecision(action=text, thought=thought, raw_response=text)

    @staticmethod
    def _extract_directive(text: str, prefix: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                return stripped[len(prefix) :].strip()
        if prefix in text:
            remainder = text.split(prefix, 1)[-1].strip()
            return remainder.splitlines()[0].strip() if remainder else ""
        return ""

    @staticmethod
    def _parse_skill_directive(directive: str) -> tuple[str, Dict[str, Any]]:
        text = directive.strip()
        if not text:
            return "memory_retrieval", {}
        if "(" not in text or not text.endswith(")"):
            return text, {}

        name, arg_text = text.split("(", 1)
        skill_name = name.strip() or "memory_retrieval"
        arg_text = arg_text[:-1].strip()
        if not arg_text:
            return skill_name, {}

        try:
            parsed = ast.parse(f"f({arg_text})", mode="eval")
            call = parsed.body
            if isinstance(call, ast.Call):
                arguments: Dict[str, Any] = {}
                for keyword in call.keywords:
                    if keyword.arg is None:
                        continue
                    try:
                        arguments[keyword.arg] = ast.literal_eval(keyword.value)
                    except Exception:
                        try:
                            arguments[keyword.arg] = ast.unparse(keyword.value)  # type: ignore[attr-defined]
                        except Exception:
                            arguments[keyword.arg] = None
                if arguments:
                    return skill_name, arguments
        except Exception:
            logger.debug("Failed to parse skill invocation arguments", exc_info=True)

        return skill_name, {"raw": arg_text}
