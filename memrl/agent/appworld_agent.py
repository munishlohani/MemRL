"""AppWorld agent variant: code-block actions over a multi-turn episode.

AppWorld sits between the two existing patterns:

- Like BCB, the action payload is a multi-line fenced Python block, so
  CustomAgent's single-line `_extract_directive` would truncate it to
  "```python". BCBAgent already solved that with a line-anchored multi-line
  extractor, and this class reuses it rather than growing a third copy.
- Unlike BCB (single step), the episode is multi-turn: each turn carries the
  running interaction history plus the stdout of the previous block, the way
  LLBAgent does.

The full raw code block is forwarded to the adapter, which hands it to
`world.execute(...)` inside the AppWorld worker. This agent never inspects or
rewrites the code -- the environment is the only thing that decides whether it
runs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from memrl.appworld_eval.prompts import (
    APPWORLD_SKILL_AWARE_PROMPT,
    APPWORLD_STRATEGIC_SELECTION_SYSTEM_PROMPT,
    APPWORLD_SYSTEM_PROMPT,
    APPWORLD_ZERO_SHOT_PROMPT,
)

from .base import AgentDecision, EnvActionDecision, SkillInvocationDecision
from .bcb_agent import BCBAgent
from .custom_agent import CustomAgent
from .prompts import STRATEGIC_SELECTION_USER_PROMPT


class AppWorldAgent(CustomAgent):
    """CustomAgent with AppWorld prompting and fenced-code-block parsing."""

    strategic_selection_system_prompt = APPWORLD_STRATEGIC_SELECTION_SYSTEM_PROMPT
    strategic_selection_user_prompt = STRATEGIC_SELECTION_USER_PROMPT

    def _build_messages(
        self,
        *,
        observation: str,
        history_messages: List[Dict[str, Any]],
        first_step: bool,
        admissible_commands: Sequence[str] = (),
        skill_budget_remaining: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        # AppWorld has no admissible-actions list -- any Python is legal, the
        # APIs decide what works. first_step is not used: the running history
        # and last output already say where the episode is.
        del first_step, admissible_commands

        skill_contract = self._render_memory_retrieval_contract()
        # Same framing as LLBAgent: an unused call is wasted, rather than
        # ALFWorld's "use it when it matters, not on every turn", which at a
        # budget of 1 only teaches the agent to hoard it.
        if skill_contract and skill_budget_remaining is not None:
            if skill_budget_remaining > 0:
                budget_note = (
                    f"You have {skill_budget_remaining} memory_retrieval call(s) available "
                    "this episode. An unused call is wasted. Use it where a past procedure "
                    "would help most."
                )
            else:
                budget_note = (
                    "Your memory_retrieval budget for this episode is used up. Act "
                    "directly for the rest of the episode."
                )
            skill_contract = f"{skill_contract}\n\n{budget_note}"

        system_content = (
            self.system_prompt.format(skill_contract=skill_contract)
            if "{skill_contract}" in self.system_prompt
            else self.system_prompt
        )
        messages = [{"role": "system", "content": system_content}]

        history_text = self._format_history_messages(history_messages)
        template = APPWORLD_SKILL_AWARE_PROMPT if skill_contract else APPWORLD_ZERO_SHOT_PROMPT
        messages.append(
            {
                "role": "user",
                "content": template.format(
                    task_description=self.task_description.strip(),
                    history=history_text,
                    observation=observation.strip() or "(no output yet)",
                ),
            }
        )
        return messages

    @staticmethod
    def _parse_decision(response: str) -> AgentDecision:
        text = (response or "").strip()
        if not text:
            return EnvActionDecision(action="", raw_response="")

        skill_directive = CustomAgent._extract_directive(text, "Skill:")
        skill_line_index = BCBAgent._first_directive_line_index(text, "Skill:")
        action_line_index = BCBAgent._first_directive_line_index(text, "Action:")
        action_directive = (
            BCBAgent._extract_multiline_directive(text, "Action:")
            if action_line_index != -1
            else ""
        )

        if skill_directive and (
            action_line_index == -1
            or (skill_line_index != -1 and skill_line_index < action_line_index)
        ):
            skill_name, arguments = CustomAgent._parse_skill_directive(skill_directive)
            return SkillInvocationDecision(
                skill_name=skill_name,
                arguments=arguments,
                raw_response=text,
            )

        if action_directive:
            return EnvActionDecision(action=action_directive, raw_response=text)

        # No Action: line at all. Forward the raw text -- the adapter strips the
        # code fence, and a bare fenced block with no directive is the common
        # slip. Inventing a no-op here would silently waste the turn instead.
        return EnvActionDecision(action=text, raw_response=text)


__all__ = ["AppWorldAgent", "APPWORLD_SYSTEM_PROMPT"]
