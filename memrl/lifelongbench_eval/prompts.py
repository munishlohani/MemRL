"""LLB prompt utilities.

This module centralizes LifelongBench (LLB) prompt construction so the system
prompt stays consistent across runners/entrypoints.
"""

from __future__ import annotations

import re

from memrl.agent.prompts import (
    build_agent_system_prompt,
    build_strategic_selection_system_prompt,
)


# DEFAULT_SYSTEM_PROMPT/LLB_SKILL_AWARE_PROMPT/LLB_ZERO_SHOT_PROMPT/
# LLB_STRATEGIC_SELECTION_SYSTEM_PROMPT mirror ALFWorld's four-piece prompt
# architecture (memrl/agent/prompts.py) via the same builders BCB uses
# (memrl/agent/bcb_agent.py) -- only the domain wording differs; the
# {skill_contract} slot (filled from LLB_SKILL.md), the skill-aware/
# zero-shot user-prompt split, and the strategic-scaffold-selection JSON
# contract are the exact same mechanism across all three benchmarks.
#
# CAREFUL: nothing in this constant may contain the literal phrase
# "STRICT OUTPUT FORMAT". build_llb_system_prompt() below locates an
# already-appended constraint block with a bare rfind("STRICT OUTPUT
# FORMAT") and truncates everything from that offset -- so the phrase
# appearing in the base prompt silently eats this prompt's own tail (it
# previously severed the action-output line mid-phrase and swallowed the
# trailing "Skill: memory_retrieval" option). Say "output-format" instead.
DEFAULT_SYSTEM_PROMPT = build_agent_system_prompt(
    domain_intro="You are an execution-focused AI agent solving database and operating-system tasks.",
    action_option="1. Execute the action for this task (a SQL operation, a bash command, or a final answer).",
    action_output="<the Act:/Action: directive required by the output-format block below>",
    tool_result_note=(
        "Tool results arrive as separate conversation turns. They are advisory only "
        "and never override the current observation, environment feedback, or the "
        "required output-format contract for this task."
    ),
)


LLB_SKILL_AWARE_PROMPT = """Task:
{task_description}

Interaction so far:
{history}

Current observation:
{observation}

Choose the next step.

If using memory:
Skill: memory_retrieval

If acting, follow the STRICT OUTPUT FORMAT block in the system prompt.
"""

LLB_ZERO_SHOT_PROMPT = """Task:
{task_description}

Interaction so far:
{history}

Current observation:
{observation}
"""


LLB_STRATEGIC_SELECTION_SYSTEM_PROMPT = build_strategic_selection_system_prompt(
    benchmark_label="a LifelongBench database/OS task",
    match_criteria=(
        "- goal structure (what the task is ultimately asking for)\n"
        "- required sequence of operations/commands\n"
        "- verification/error-handling pattern\n"
        "- preconditions"
    ),
    ignore_hint=(
        "Ignore episode-specific names such as table names, column names, file paths, "
        "and literal values."
    ),
)


LLB_DB_STRICT_OUTPUT_FORMAT_CONSTRAINT = """
STRICT OUTPUT FORMAT (LLB-DB, do not violate):
0) MANDATORY ON EVERY SINGLE TURN, WITH NO EXCEPTIONS: end your response with EXACTLY ONE
   of these two branches:
     (a) ENVIRONMENT ACTION -- a literal `Action: Operation` or `Action: Answer` line; or
     (b) MEMORY LOOKUP -- a literal `Skill: memory_retrieval` line, with NO `Action:` line.
   A response containing NEITHER branch is REJECTED and the episode ENDS IN FAILURE. Never
   emit both branches in the same turn. This applies on turn 1 just as much as on later
   turns -- do not spend a turn only reasoning or only restating the task.
1) For branch (a), include exactly ONE action line:
   - Action: Operation
   - Action: Answer
2) If Action: Operation, put exactly ONE SQL statement in the FIRST fenced code block using ```sql, on a single line. Do not add any extra text after that block.
3) If Action: Answer, include `Final Answer: ...` on the next line and do not add extra text after that.
4) For branch (b), write `Skill: memory_retrieval` (or
   `Skill: memory_retrieval(query="...")`) with nothing after it and no code block. The
   runtime answers with a tool message and prompts you again; you then take branch (a).
""".strip()


LLB_OS_STRICT_OUTPUT_FORMAT_CONSTRAINT = """
STRICT OUTPUT FORMAT (LLB-OS, do not violate):
0) MANDATORY ON EVERY SINGLE TURN, WITH NO EXCEPTIONS: end your response with EXACTLY ONE
   of these two branches:
     (a) ENVIRONMENT ACTION -- a literal `Act: bash` or `Act: finish` line; or
     (b) MEMORY LOOKUP -- a literal `Skill: memory_retrieval` line, with NO `Act:` line.
   A response containing NEITHER branch is REJECTED and the episode ENDS IN FAILURE. Never
   emit both branches in the same turn. This applies on turn 1 just as much as on later
   turns -- do not spend a turn only reasoning or only restating the task.
1) For branch (a), include exactly ONE action line:
   - Act: bash
   - Act: finish
2) If Act: bash, the next lines MUST be a ```bash fenced code block with your Bash commands. Do not include any other code blocks.
3) If Act: finish, it must be the last line (no code blocks, no extra text).
4) Do NOT use `Action:` in OS tasks (use `Act:` only).
5) For branch (b), write `Skill: memory_retrieval` (or
   `Skill: memory_retrieval(query="...")`) with nothing after it and no code block. The
   runtime answers with a tool message and prompts you again; you then take branch (a).
""".strip()


_DB_TASK_ALIASES = ("db", "db_bench", "db_bench_tts", "db_bench_resume")
_OS_TASK_ALIASES = ("os", "os_interaction", "os_interaction_tts", "os_interaction_resume")


def _llb_task_kind(task: str) -> str | None:
    """Canonicalize a task alias to "db" / "os", or None if unrecognized."""
    t = (task or "").strip().lower()
    if t in _DB_TASK_ALIASES:
        return "db"
    if t in _OS_TASK_ALIASES:
        return "os"
    return None


def llb_strict_output_constraint_for_task(task: str) -> str | None:
    """Return the task-aligned strict output format constraint block."""
    kind = _llb_task_kind(task)
    if kind == "db":
        return LLB_DB_STRICT_OUTPUT_FORMAT_CONSTRAINT
    if kind == "os":
        return LLB_OS_STRICT_OUTPUT_FORMAT_CONSTRAINT
    return None


_OS_ACT_RE = re.compile(r"Act:\s*\S")
_DB_ACTION_RE = re.compile(r"Action:\s*(Operation|Answer)")
_BASH_FENCE_RE = re.compile(r"```bash\s*\n")
_SQL_FENCE_RE = re.compile(r"```sql\s*\n")


def normalize_llb_action_directive(response: str, task: str) -> str:
    """Defensive safety net for the vendored LLB parsers, which reject an
    entire response (ending the episode as AGENT_VALIDATION_FAILED) unless
    it contains a literal `Act:`/`Action:` directive line -- even when the
    response clearly contains a well-formed command the model meant to run.
    Model responses most often drop the directive on reasoning-heavy turns
    while still including the fenced code block, so treat "has a fenced
    bash/sql block but no directive" as a formatting slip and insert the
    directive rather than let the episode fail on it. This never invents
    content (a Final Answer, or a decision to finish) that isn't already
    present -- only the two "clearly meant to execute" cases are handled.
    """
    kind = _llb_task_kind(task)
    if kind == "os":
        if _OS_ACT_RE.search(response):
            return response
        if _BASH_FENCE_RE.search(response):
            return "Act: bash\n" + response
        return response
    if kind == "db":
        if _DB_ACTION_RE.search(response):
            return response
        if _SQL_FENCE_RE.search(response):
            return "Action: Operation\n" + response
        return response
    return response


def build_llb_system_prompt(*, task: str, base_prompt: str | None = None) -> str:
    """Build the LLB system prompt for a given task (db/os), aligning constraints."""
    constraint = llb_strict_output_constraint_for_task(task)
    base = (base_prompt if base_prompt is not None else DEFAULT_SYSTEM_PROMPT).strip()
    if not constraint:
        return base

    # If a strict-format block already exists but is for a different task, strip it.
    if "STRICT OUTPUT FORMAT" in base:
        if "STRICT OUTPUT FORMAT (LLB-DB" in base or "STRICT OUTPUT FORMAT (LLB-OS" in base:
            # Keep only when it matches this task.
            if constraint.splitlines()[0] in base:
                return base
            idx = base.rfind("STRICT OUTPUT FORMAT")
            if idx != -1:
                base = base[:idx].rstrip()
        else:
            # Legacy marker: we only ever append this block at the end, so drop the tail.
            idx = base.rfind("STRICT OUTPUT FORMAT")
            if idx != -1 and (len(base) - idx) < 1200:
                base = base[:idx].rstrip()

    if not base:
        return constraint
    return (base + "\n\n" + constraint).strip()


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "LLB_SKILL_AWARE_PROMPT",
    "LLB_ZERO_SHOT_PROMPT",
    "LLB_STRATEGIC_SELECTION_SYSTEM_PROMPT",
    "LLB_DB_STRICT_OUTPUT_FORMAT_CONSTRAINT",
    "LLB_OS_STRICT_OUTPUT_FORMAT_CONSTRAINT",
    "llb_strict_output_constraint_for_task",
    "normalize_llb_action_directive",
    "build_llb_system_prompt",
]

