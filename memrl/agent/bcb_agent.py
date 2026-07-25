"""BigCodeBench agent variant: dedicated prompting + multi-line Action: parsing.

Two things CustomAgent gets wrong for BCB:

1. Prompting: CustomAgent._build_messages renders the shared, ALFWorld-authored
   prompts.SKILL_AWARE_PROMPT/ZERO_SHOT_PROMPT, which reference an "admissible
   actions list" BCB doesn't have (BCB never supplies admissible_commands, so
   the user turn ends up saying "Admissible actions: (none available)" and
   "Action: <command copied verbatim from the admissible actions list>" --
   directly contradicting BCB_SYSTEM_PROMPT's fenced-python-block format).
   BCBAgent overrides _build_messages with a minimal, BCB-only user message
   (task prompt + interaction-so-far only if a skill round-trip happened) --
   closer to the original (pre-agentic) BCBRunner's single "system instructions
   + bare task prompt" shape than to ALFWorld's templated one, with no
   few-shot section (few-shot was never wired for BCB in the first place --
   CustomAgent.__init__ already discards few_shot_examples unconditionally).

2. Parsing: CustomAgent._parse_decision extracts "Action:"/"Skill:" via a raw
   whole-text substring search (text.split("Action:", 1), text.find(...)),
   not line-anchored the way its own Thought:/Skill: single-line extraction
   is. A BigCodeBench solution after "Action:" is a multi-line fenced Python
   block (single-line extraction would truncate it), and the raw substring
   search means an incidental "Action:"/"Skill:" appearing inside the model's
   own preceding prose corrupts extraction. _parse_decision calls
   CustomAgent._extract_directive by hard-coded class name (not
   polymorphically), so overriding _extract_directive alone would never run;
   _parse_decision itself must be overridden here.

BCB_SYSTEM_PROMPT doesn't ask for a Thought: reasoning preamble either (the
original BCBRunner never did) -- the response is just Action: or Skill:.
The skill invocation itself stays agent-driven (the model decides whether to
retrieve at all, same as ALFWorld), but run_bcb.py caps EpisodeRunner's
max_skill_invocations at 1 for BCB (vs. the module default of 3), since BCB's
whole episode is a single action budget (max_steps=1) -- one retrieval turn
plus one submission turn, not several retrieval attempts before ever
answering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .base import AgentDecision, EnvActionDecision, SkillInvocationDecision
from .custom_agent import CustomAgent


BCB_SYSTEM_PROMPT = """You are an expert Python programmer solving BigCodeBench coding tasks.

For each turn, choose exactly one branch:
1. Direct submission:
   Action:
   ```python
   <your complete solution, one Python code block, nothing else>
   ```
2. Memory skill invocation:
   Skill: memory_retrieval

If you invoke the skill, the runtime will append a tool message with retrieved context and ask you again. Do not emit both a skill call and a code submission in the same turn. You may invoke the skill at most once per task -- use your other turn to submit code.

You may receive retrieved memory context with past experiences from similar problems -- references for learning, not guaranteed solutions:
- [MEMORY TYPE] SUCCESS_PROCEDURE: a successful approach from a similar task -- learn the implementation pattern.
- [MEMORY TYPE] FAILURE_REFLECTION: a failed attempt with lessons -- avoid similar mistakes.
Use them as inspiration, but always analyze the current task independently.

Hard constraints for BigCodeBench:
- Do NOT change the required function signature, return type, or required exception types/messages.
- Do NOT wrap specific exceptions into generic ones; keep the exact exception class and message if specified.
- Import every module you use; remove unused imports; do not rely on implicit imports.
- Avoid broad try/except (e.g., `except Exception`) unless the task explicitly requires it.
- Avoid any network calls or extra file I/O beyond what the task specifies.
- Keep code deterministic: no randomness, time-based logic, or unnecessary logging.
- Your Action must contain the ENTIRE solution as one fenced ```python code block immediately after the "Action:" line -- no prose inside the fence, nothing after it.

Your response should use one of the following formats:

Action:
```python
<code>
```

Skill: memory_retrieval"""


class BCBAgent(CustomAgent):
    """CustomAgent with BCB-only prompting and multi-line Action: parsing."""

    def _build_messages(
        self,
        *,
        observation: str,
        history_messages: List[Dict[str, Any]],
        first_step: bool,
        admissible_commands: Sequence[str] = (),
    ) -> List[Dict[str, str]]:
        # observation duplicates task_description for BCB (the adapter sets
        # both to the same prompt string) and there is no admissible-actions
        # list, so neither is used here -- unlike CustomAgent's ALFWorld
        # rendering, which needs both. first_step is also not a useful
        # signal: it is computed once per environment STEP and held constant
        # across every inner skill-retry turn within that one step (see
        # EpisodeRunner._resolve_agent_turn), so it can't distinguish "first
        # LLM call" from "call after a skill invocation" the way checking
        # history_messages directly can.
        del observation, first_step, admissible_commands

        messages = [{"role": "system", "content": self.system_prompt}]

        user_parts = [self.task_description.strip()]
        if history_messages:
            # Only present after this episode already invoked the memory
            # skill at least once (BCB's max_steps=1 means history_messages
            # is otherwise always empty going into the first call) -- carries
            # the tool message with retrieved memories forward.
            user_parts.append(
                f"Interaction so far:\n{self._format_history_messages(history_messages)}"
            )
        messages.append({"role": "user", "content": "\n\n".join(user_parts)})
        return messages

    @staticmethod
    def _parse_decision(response: str) -> AgentDecision:
        # No Thought: extraction -- BCB_SYSTEM_PROMPT doesn't ask for a
        # reasoning preamble (the original BCBRunner never did either), so
        # AgentDecision.thought stays at its "" default for BCB.
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
            return EnvActionDecision(
                action=action_directive,
                raw_response=text,
            )

        return EnvActionDecision(action=text, raw_response=text)

    @staticmethod
    def _first_directive_line_index(text: str, prefix: str) -> int:
        """Index of the first LINE whose stripped content starts with prefix.

        Line-anchored (unlike a raw text.find(prefix)) so an incidental
        "Action:"/"Skill:" appearing inside the model's own Thought: prose
        doesn't get mistaken for the real directive.
        """
        for index, line in enumerate(text.splitlines()):
            if line.strip().startswith(prefix):
                return index
        return -1

    @staticmethod
    def _extract_multiline_directive(text: str, prefix: str) -> str:
        """Like CustomAgent._extract_directive, but keeps everything from the
        matching line to the end of the text instead of just that one line --
        Action:'s payload is a multi-line fenced code block, not a single
        short command. Still line-anchored: the prefix must start a stripped
        line, matching _first_directive_line_index's detection.
        """
        lines = text.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(prefix):
                remainder_of_line = stripped[len(prefix):].strip()
                rest = lines[index + 1 :]
                if remainder_of_line:
                    return "\n".join([remainder_of_line, *rest]).strip()
                return "\n".join(rest).strip()
        return ""


__all__ = ["BCBAgent", "BCB_SYSTEM_PROMPT"]
