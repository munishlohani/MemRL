"""LifelongBench (LLB) agent variant: DB/OS task-type-aware prompting + parsing.

LLB covers two structurally different task grammars from the vendored
LifelongAgentBench library (3rdparty/LifelongAgentBench): DBBench expects
"Action: Operation"/"Action: Answer" (memrl/lifelongbench_eval/prompts.py's
LLB_DB_STRICT_OUTPUT_FORMAT_CONSTRAINT); OSInteraction expects "Act: bash"/
"Act: finish" (LLB_OS_STRICT_OUTPUT_FORMAT_CONSTRAINT). Both vendored Task
classes do their OWN parsing of the raw response text inside
task_obj.interact() (db_bench/task.py::_parse_agent_response,
os_interaction/task.py::_parse_agent_response) -- LLBAgent forwards the
model's full raw response as the env action untouched rather than
re-extracting the SQL/bash payload itself, so there is exactly one place
(the vendored parser) that decides what's valid, instead of two parsers
that could silently drift apart. _parse_decision here only needs to decide
Skill: memory_retrieval vs. a genuine env turn.

Unlike BCB's pre-agentic-style "system instructions + bare task prompt"
(single-step, max_steps=1), LLB is multi-turn like ALFWorld -- the user
message carries the running interaction history and current observation
every turn, built from EpisodeRunner's own EpisodeHistory (LLBEpisodeEnvAdapter
never exposes the vendored library's internal Session.chat_history directly
to the agent; that separation is the whole point of the EpisodeEnvAdapter
abstraction).

Task-type-aware system prompt: memrl/lifelongbench_eval/prompts.py's
build_llb_system_prompt() already appends the correct DB/OS strict
output-format block and is reused verbatim. LLB's old
always-retrieve-upfront helpers (memory_context.py/sanitize.py, which
bucketed memories into the system prompt before the episode started) were
deleted along with that design: memory retrieval here is agent-invoked
(Skill: memory_retrieval, same as ALFWorld/BCB), so retrieved content
arrives as a tool message in conversation history rather than as a
pre-formatted system-prompt block.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from memrl.lifelongbench_eval.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    LLB_SKILL_AWARE_PROMPT,
    LLB_STRATEGIC_SELECTION_SYSTEM_PROMPT,
    LLB_ZERO_SHOT_PROMPT,
    build_llb_system_prompt,
)

from .base import AgentDecision, EnvActionDecision, SkillInvocationDecision
from .custom_agent import CustomAgent
from .prompts import STRATEGIC_SELECTION_USER_PROMPT


# DEFAULT_SYSTEM_PROMPT already covers the Skill: memory_retrieval mechanics
# (agent-invoked, not automatic) -- this alias just names it for run_llb.py's
# call site.
LLB_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


class LLBAgent(CustomAgent):
    """CustomAgent with LLB (DB/OS) task-aware prompting and parsing."""

    strategic_selection_system_prompt = LLB_STRATEGIC_SELECTION_SYSTEM_PROMPT
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
        # LLB has no admissible-actions list (unlike ALFWorld); first_step
        # isn't a useful signal here either -- the running history/current
        # observation fields already carry everything needed each turn.
        del first_step, admissible_commands

        skill_contract = self._render_memory_retrieval_contract()
        # Budget note rides along with the contract in the SYSTEM message,
        # like CustomAgent does for ALFWorld. The wording deliberately
        # DIVERGES from ALFWorld's "use it when it matters, not on every
        # turn": at skill_budget_per_episode=1 that phrasing is vacuous (it
        # cannot be used on every turn) and functions purely as a hoarding
        # signal -- the agent saves its single call for a moment that never
        # arrives and the episode ends with the budget unspent.
        if skill_contract and skill_budget_remaining is not None:
            if skill_budget_remaining > 0:
                budget_note = (
                    f"You have {skill_budget_remaining} memory_retrieval call(s) available "
                    "this episode. An unused call is wasted -- there is no benefit to "
                    "ending the episode with budget left over, so spend it on the step "
                    "where a past procedure would help most."
                )
            else:
                budget_note = (
                    "Your memory_retrieval budget for this episode is used up. Act "
                    "directly for the rest of the episode."
                )
            skill_contract = f"{skill_contract}\n\n{budget_note}"
        system_content = build_llb_system_prompt(task=self.task_type, base_prompt=self.system_prompt)
        if "{skill_contract}" in system_content:
            system_content = system_content.format(skill_contract=skill_contract)
        messages = [{"role": "system", "content": system_content}]

        history_text = self._format_history_messages(history_messages)
        template = LLB_SKILL_AWARE_PROMPT if skill_contract else LLB_ZERO_SHOT_PROMPT
        messages.append(
            {
                "role": "user",
                "content": template.format(
                    task_description=self.task_description.strip(),
                    history=history_text,
                    observation=observation.strip(),
                ),
            }
        )
        return messages

    def _parse_decision(self, response: str) -> AgentDecision:
        text = (response or "").strip()
        if not text:
            return EnvActionDecision(action="", raw_response="")

        skill_directive = CustomAgent._extract_directive(text, "Skill:")
        skill_line_index = LLBAgent._first_directive_line_index(text, "Skill:")
        action_prefix = "Act:" if self._is_os_task() else "Action:"
        action_line_index = LLBAgent._first_directive_line_index(text, action_prefix)

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

        # Forward the FULL raw response as the env action, untouched -- the
        # vendored Task's own _interact()/_parse_agent_response is the
        # authoritative parser for the DB/OS grammar; re-extracting the
        # SQL/bash payload here would duplicate (and risk diverging from)
        # that logic.
        return EnvActionDecision(action=text, raw_response=text)

    def _is_os_task(self) -> bool:
        return str(self.task_type or "").strip().lower().startswith("os")

    @staticmethod
    def _first_directive_line_index(text: str, prefix: str) -> int:
        """Index of the first LINE whose stripped content starts with prefix.

        Line-anchored (unlike a raw text.find(prefix)) so an incidental
        "Action:"/"Act:"/"Skill:" appearing inside the model's own preceding
        prose doesn't get mistaken for the real directive.
        """
        for index, line in enumerate(text.splitlines()):
            if line.strip().startswith(prefix):
                return index
        return -1


__all__ = ["LLBAgent", "LLB_SYSTEM_PROMPT"]
