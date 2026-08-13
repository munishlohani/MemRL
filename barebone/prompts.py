"""Standalone system prompts for the barebone (no-memory) runners.

Covers all four benchmarks -- ALFWorld, BigCodeBench, LifelongBench, and AppWorld.

Plain LLM baseline: no ReAct, no memory, no skill invocation, no retrieval
mechanism. Exactly two messages per turn (system + user) -- the user
message is built fresh each call from Task/Interaction-history/Current
observation/Admissible actions fields (see agent.py), not an accumulating
chat message list. "Interaction history so far" is rendered as plain text
inside that single user message.

Each benchmark's system prompt opens by stating the domain the agent is
operating in, then the one hard output rule. Keep that shape when editing:
the barebone baseline is compared head-to-head against the memory-enabled
agents, so a prompt that is asymmetrically weaker than its siblings turns
a measured memory effect into a prompt-quality confound.
"""

BAREBONE_ALFWORLD_SYSTEM_PROMPT = (
    "You are controlling a text-based ALFWorld environment. Choose the "
    "NEXT action as ONE admissible command string. Output only the "
    "command, copied verbatim from the admissible list."
)

BAREBONE_ALFWORLD_USER_TEMPLATE = (
    "Task: {objective}\n"
    "Interaction history so far: {history}\n"
    "Current observation: {current_obs}\n"
    "Admissible actions: {admissible}\n"
    "Action:"
)


# BigCodeBench is a single-step task (submit code, evaluate, done) -- there
# is no interaction history or admissible-actions concept, so unlike the
# ALFWorld prompt above there is no separate user template: the user
# message is just the task prompt itself (see agent.py).
BAREBONE_BCB_SYSTEM_PROMPT = (
    "Generate clean, correct Python code.\n\n"
    "Respond with your complete solution as a single fenced ```python code block, nothing else."
)


# LifelongBench (LLB) is multi-turn like ALFWorld (running interaction
# history + current observation each turn), but has no admissible-actions
# list -- DB/OS actions are free-form SQL/bash, not a fixed command set.
#
# The output-format blocks below are defined HERE rather than imported from
# memrl.lifelongbench_eval.prompts. That module's blocks describe a second
# branch, `Skill: memory_retrieval`, because the memory-augmented agent can
# take it. This runner has no skill contract and no retrieval mechanism at
# all: run_llb_barebone.py forwards the model's raw response straight to the
# vendored parser, so a `Skill:` line would arrive with no Act:/Action:
# directive and fail the episode outright.
#
# The tradeoff of duplicating is drift: this grammar must keep matching the
# vendored Task.interact() parser (Act:/Action: + a fenced bash/sql block).
# tests/test_llb_prompts.py pins that, so drift fails the suite rather than
# silently corrupting the baseline.
BAREBONE_LLB_SYSTEM_PROMPT_BASE = (
    "You are an execution-focused AI agent solving database and "
    "operating-system tasks. Take exactly one action per turn."
)

BAREBONE_LLB_OS_OUTPUT_FORMAT = """
STRICT OUTPUT FORMAT (LLB-OS, do not violate):
0) MANDATORY ON EVERY SINGLE TURN, WITH NO EXCEPTIONS: your response MUST contain a literal
   `Act: bash` or `Act: finish` line. A response without one is REJECTED and the episode
   ENDS IN FAILURE. This applies on turn 1 just as much as on later turns -- do not spend a
   turn only reasoning or only restating the task.
1) Include exactly ONE action line:
   - Act: bash
   - Act: finish
2) If Act: bash, the next lines MUST be a ```bash fenced code block with your Bash commands. Do not include any other code blocks.
3) If Act: finish, it must be the last line (no code blocks, no extra text).
4) Do NOT use `Action:` in OS tasks (use `Act:` only).
""".strip()

BAREBONE_LLB_DB_OUTPUT_FORMAT = """
STRICT OUTPUT FORMAT (LLB-DB, do not violate):
0) MANDATORY ON EVERY SINGLE TURN, WITH NO EXCEPTIONS: your response MUST contain a literal
   `Action: Operation` or `Action: Answer` line. A response without one is REJECTED and the
   episode ENDS IN FAILURE. This applies on turn 1 just as much as on later turns -- do not
   spend a turn only reasoning or only restating the task.
1) Include exactly ONE action line:
   - Action: Operation
   - Action: Answer
2) If Action: Operation, put exactly ONE SQL statement in the FIRST fenced code block using ```sql, on a single line. Do not add any extra text after that block.
3) If Action: Answer, include `Final Answer: ...` on the next line and do not add extra text after that.
""".strip()

_BAREBONE_LLB_DB_ALIASES = ("db", "db_bench", "db_bench_tts", "db_bench_resume")
_BAREBONE_LLB_OS_ALIASES = ("os", "os_interaction", "os_interaction_tts", "os_interaction_resume")


def barebone_llb_output_format(task: str) -> str | None:
    """Task-aligned output-format block for the no-memory LLB runner."""
    t = (task or "").strip().lower()
    if t in _BAREBONE_LLB_DB_ALIASES:
        return BAREBONE_LLB_DB_OUTPUT_FORMAT
    if t in _BAREBONE_LLB_OS_ALIASES:
        return BAREBONE_LLB_OS_OUTPUT_FORMAT
    return None


BAREBONE_LLB_USER_TEMPLATE = (
    "Task:\n{objective}\n\n"
    "Interaction history so far:\n{history}\n\n"
    "Current observation:\n{current_obs}"
)
BAREBONE_APPWORLD_SYSTEM_PROMPT = (
    "You are completing a task by calling app APIs from Python code.\n\n"
    "Respond with exactly:\nAction:\n```python\n<python code to execute this turn>\n```"
)

BAREBONE_APPWORLD_USER_TEMPLATE = (
    "Task:\n{objective}\n\n"
    "Interaction history so far:\n{history}\n\n"
    "Output of your last code block:\n{current_obs}"
)


__all__ = [
    "BAREBONE_ALFWORLD_SYSTEM_PROMPT",
    "BAREBONE_ALFWORLD_USER_TEMPLATE",
    "BAREBONE_BCB_SYSTEM_PROMPT",
    "BAREBONE_LLB_SYSTEM_PROMPT_BASE",
    "BAREBONE_LLB_OS_OUTPUT_FORMAT",
    "BAREBONE_LLB_DB_OUTPUT_FORMAT",
    "barebone_llb_output_format",
    "BAREBONE_LLB_USER_TEMPLATE",
    "BAREBONE_APPWORLD_SYSTEM_PROMPT",
    "BAREBONE_APPWORLD_USER_TEMPLATE",
]
