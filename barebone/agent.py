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

from prompts import BAREBONE_ALFWORLD_SYSTEM_PROMPT, BAREBONE_ALFWORLD_USER_TEMPLATE


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


__all__ = ["BarebonAgent"]
