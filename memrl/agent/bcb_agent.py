"""BigCodeBench agent variant: CustomAgent with multi-line Action: parsing.

CustomAgent._parse_decision extracts "Action:" via _extract_directive, which
is single-line only -- correct for ALFWorld's short verb commands, but a
BigCodeBench solution after "Action:" is a multi-line fenced Python block
that would otherwise be silently truncated to its first line.
_parse_decision calls CustomAgent._extract_directive by hard-coded class
name (not polymorphically), so overriding _extract_directive alone would
never run; _parse_decision itself must be overridden here.
"""

from __future__ import annotations

from .base import AgentDecision, EnvActionDecision, SkillInvocationDecision
from .custom_agent import CustomAgent


BCB_SYSTEM_PROMPT = """You are an expert Python programmer solving BigCodeBench coding tasks.

For each turn, choose exactly one branch:
1. Direct submission:
   Thought: <your reasoning>
   Action:
   ```python
   <your complete solution, one Python code block, nothing else>
   ```
2. Memory skill invocation:
   Thought: <your reasoning>
   Skill: memory_retrieval

If you invoke the skill, the runtime will append a tool message with retrieved context and ask you again. Do not emit both a skill call and a code submission in the same turn.

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

Thought: <your thoughts>
Action:
```python
<code>
```

Thought: <your thoughts>
Skill: memory_retrieval"""


class BCBAgent(CustomAgent):
    """CustomAgent whose Action: directive may span multiple lines."""

    @staticmethod
    def _parse_decision(response: str) -> AgentDecision:
        text = (response or "").strip()
        if not text:
            return EnvActionDecision(action="", raw_response="")

        skill_directive = CustomAgent._extract_directive(text, "Skill:")
        thought = CustomAgent._extract_directive(text, "Thought:")
        action_directive = text.split("Action:", 1)[-1].strip() if "Action:" in text else ""

        skill_index = text.find("Skill:")
        action_index = text.find("Action:")

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


__all__ = ["BCBAgent", "BCB_SYSTEM_PROMPT"]
