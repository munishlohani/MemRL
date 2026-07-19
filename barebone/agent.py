"""Minimal ReAct agent for the barebone ALFWorld runner.

No memory, no tools, no skill invocation of any kind -- this module does
not import anything from memrl.agent/memrl.skills/memrl.service, so it is
architecturally incapable of touching the skill-memory system, not just
prompted not to.
"""

from __future__ import annotations

from typing import Any, Dict, List

from prompts import BAREBONE_ALFWORLD_SYSTEM_PROMPT


class BarebonAgent:
    """Plain Thought:/Action: agent, one instance per parallel episode slot."""

    def __init__(self, llm_provider: Any, *, system_prompt: str = BAREBONE_ALFWORLD_SYSTEM_PROMPT) -> None:
        self.llm = llm_provider
        self.system_prompt = system_prompt
        self._history: List[Dict[str, str]] = []

    def reset(self) -> None:
        self._history = []

    def act(self, observation: str) -> str:
        user_turn = {"role": "user", "content": f"Observation: {observation}"}
        messages = [{"role": "system", "content": self.system_prompt}, *self._history, user_turn]

        response = self.llm.generate(messages=messages,extra_body={"chat_template_kwargs": {"enable_thinking": False}} )
        action, thought = self._parse_response(response)

        self._history.append(user_turn)
        self._history.append({"role": "assistant", "content": response or f"Action: {action}"})
        return action

    @staticmethod
    def _parse_response(response: str) -> tuple[str, str]:
        text = (response or "").strip()
        if not text:
            return "look", ""

        action = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("action:"):
                action = stripped[len("action:") :].strip()

        if not action:
            # No "Action:" line found at all -- fall back to the whole
            # response so a malformed reply doesn't silently no-op forever.
            action = text.splitlines()[-1].strip()

        return action or "look"


__all__ = ["BarebonAgent"]
