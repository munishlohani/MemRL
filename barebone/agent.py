"""Plain LLM agent for the barebone ALFWorld runner.

No memory, no tools, no skill invocation of any kind -- this module does
not import anything from memrl.agent/memrl.skills/memrl.service, so it is
architecturally incapable of touching the skill-memory system, not just
prompted not to.

Exactly two messages are sent per turn (system + user) -- not a growing
chat message list. "Interaction history so far" is rendered as plain text
inside the single user message, rebuilt fresh from the task description,
accumulated history text, current observation, and admissible actions
each call. There is no tag to strictly require in the response -- the raw
stripped model output (optionally prefixed "Action:") is the action.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from memrl.bigcodebench_eval.bcb_adapter import extract_code_from_response

from prompts import (
    BAREBONE_ALFWORLD_SYSTEM_PROMPT,
    BAREBONE_ALFWORLD_USER_TEMPLATE,
    BAREBONE_APPWORLD_SYSTEM_PROMPT,
    BAREBONE_APPWORLD_USER_TEMPLATE,
    BAREBONE_BCB_SYSTEM_PROMPT,
    BAREBONE_LLB_SYSTEM_PROMPT_BASE,
    BAREBONE_LLB_USER_TEMPLATE,
    barebone_llb_output_format,
)


class BarebonAgent:
    """Action-selection agent, one instance per parallel episode slot."""

    def __init__(self, llm_provider: Any, *, system_prompt: str = BAREBONE_ALFWORLD_SYSTEM_PROMPT) -> None:
        self.llm = llm_provider
        self.system_prompt = system_prompt
        self.task_description = ""
        self._history_text = ""
        self._last_action: Optional[str] = None

    def reset(self, task_description: str = "") -> None:
        self.task_description = task_description
        self._history_text = ""
        self._last_action = None

    def act(self, observation: str, admissible_commands: Sequence[str] = ()) -> str:
        # The result of the PREVIOUS action (this turn's observation) is
        # what completes that step's history entry -- recorded here,
        # before building this turn's prompt, since act() only learns the
        # outcome of its last chosen action on the following call.
        if self._last_action is not None:
            self._history_text += f"Action: {self._last_action}\nObservation: {observation}\n"

        user_content = BAREBONE_ALFWORLD_USER_TEMPLATE.format(
            objective=self.task_description,
            history=self._history_text.strip() or "(none yet)",
            current_obs=observation,
            admissible=", ".join(admissible_commands) if admissible_commands else "(none available)",
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = self.llm.generate(messages=messages)
        action = self._parse_response(response)
        self._last_action = action
        return action

    @staticmethod
    def _parse_response(response: str) -> str:
        text = (response or "").strip()
        if not text:
            return "look"

        action = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("action:"):
                action = stripped[len("action:") :].strip()

        if not action:
            # No "Action:" prefix in the response -- the prompt's trailing
            # "Action:" cue already supplies it, so the raw response is
            # expected to just be the command itself.
            action = text.splitlines()[-1].strip()

        return action or "look"


class BarebonBCBAgent:
    """Single-call BigCodeBench agent: no memory, no history, no loop.

    BigCodeBench is a single-step task (submit code, evaluate, done), so
    unlike BarebonAgent there is no per-episode state to track across
    calls -- one task prompt in, one code submission out.
    """

    def __init__(self, llm_provider: Any, *, system_prompt: str = BAREBONE_BCB_SYSTEM_PROMPT) -> None:
        self.llm = llm_provider
        self.system_prompt = system_prompt

    def act(self, task_prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        response = self.llm.generate(messages=messages)
        return extract_code_from_response(response)


class BarebonLLBAgent:
    """LifelongBench (LLB) agent: no memory, multi-turn like BarebonAgent,
    but with no admissible-actions list (DB/OS actions are free-form
    SQL/bash, not a fixed command set) and no action-line extraction.

    The vendored Task.interact() does its OWN parsing of the raw response
    text (its "STRICT OUTPUT FORMAT" grammar -- Action:/Act: + a fenced
    sql/bash block), so act() returns the model's full response untouched,
    the same way memrl.agent.llb_agent.LLBAgent forwards it for the
    memory-augmented pipeline: there is exactly one place that decides
    what's a valid action (the vendored parser), not two that could drift.

    The output-format block comes from barebone/prompts.py, NOT from
    memrl.lifelongbench_eval.prompts: this agent has no skill contract, so
    it must never be shown the `Skill: memory_retrieval` branch that the
    memory-augmented blocks describe.
    """

    def __init__(self, llm_provider: Any, *, task: str) -> None:
        self.llm = llm_provider
        constraint = barebone_llb_output_format(task)
        self.system_prompt = (
            f"{BAREBONE_LLB_SYSTEM_PROMPT_BASE}\n\n{constraint}" if constraint else BAREBONE_LLB_SYSTEM_PROMPT_BASE
        )
        self.task_description = ""
        self._history_text = ""
        self._last_action: Optional[str] = None

    def reset(self, task_description: str = "") -> None:
        self.task_description = task_description
        self._history_text = ""
        self._last_action = None

    def act(self, observation: str) -> str:
        # Same convention as BarebonAgent: the result of the PREVIOUS
        # action (this turn's observation) completes that step's history
        # entry, recorded before building this turn's prompt.
        if self._last_action is not None:
            self._history_text += f"{self._last_action}\nObservation: {observation}\n"

        user_content = BAREBONE_LLB_USER_TEMPLATE.format(
            objective=self.task_description,
            history=self._history_text.strip() or "(none yet)",
            current_obs=observation,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = self.llm.generate(messages=messages)
        action = (response or "").strip()
        self._last_action = action
        return action


class BarebonAppWorldAgent:
    """AppWorld agent: no memory, multi-turn, fenced-Python-code actions.

    Same history-accumulation convention as BarebonAgent/BarebonLLBAgent:
    the result of the PREVIOUS action (this turn's observation) completes
    that step's history entry, recorded before building this turn's
    prompt. Unlike BarebonBCBAgent's single call, AppWorld is a running
    multi-turn episode (code execution is stateful across turns), so
    history has to persist across act() calls the same way it does for
    ALFWorld/LLB.

    Code is pulled out with extract_code_from_response, the same fenced
    ```python block regex BCB uses -- AppWorld's own action format
    ("Action:\\n```python ... ```") is that same shape with a directive
    line the regex simply ignores.
    """

    def __init__(self, llm_provider: Any, *, system_prompt: str = BAREBONE_APPWORLD_SYSTEM_PROMPT) -> None:
        self.llm = llm_provider
        self.system_prompt = system_prompt
        self.task_description = ""
        self._history_text = ""
        self._last_action: Optional[str] = None

    def reset(self, task_description: str = "") -> None:
        self.task_description = task_description
        self._history_text = ""
        self._last_action = None

    def act(self, observation: str) -> str:
        if self._last_action is not None:
            self._history_text += f"Action:\n```python\n{self._last_action}\n```\nOutput: {observation}\n"

        user_content = BAREBONE_APPWORLD_USER_TEMPLATE.format(
            objective=self.task_description,
            history=self._history_text.strip() or "(none yet)",
            current_obs=observation.strip() or "(no output yet)",
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = self.llm.generate(messages=messages)
        code = extract_code_from_response(response)
        self._last_action = code
        return code


__all__ = ["BarebonAgent", "BarebonBCBAgent", "BarebonLLBAgent", "BarebonAppWorldAgent"]
