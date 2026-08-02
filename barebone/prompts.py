"""Standalone system prompt for the barebone ALFWorld runner.

Plain LLM baseline: no ReAct, no memory, no skill invocation, no retrieval
mechanism. Exactly two messages per turn (system + user) -- the user
message is built fresh each call from Task/Interaction-history/Current
observation/Admissible actions fields (see agent.py), not an accumulating
chat message list. "Interaction history so far" is rendered as plain text
inside that single user message.
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
# The STRICT OUTPUT FORMAT block is reused verbatim from
# memrl.lifelongbench_eval.prompts (not duplicated here): it's the exact
# grammar contract the vendored Task.interact()'s own parser depends on
# (Action:/Act: + a fenced sql/bash block), so getting it wrong isn't a
# style choice, it's a correctness bug. Only the memory-context section of
# that module's DEFAULT_SYSTEM_PROMPT is dropped -- this is the no-memory
# baseline, so there is never anything to retrieve.
BAREBONE_LLB_SYSTEM_PROMPT_BASE = (
 "Take exactly one "
    "action per turn."
)

BAREBONE_LLB_USER_TEMPLATE = (
    "Task:\n{objective}\n\n"
    "Interaction history so far:\n{history}\n\n"
    "Current observation:\n{current_obs}"
)


__all__ = [
    "BAREBONE_ALFWORLD_SYSTEM_PROMPT",
    "BAREBONE_ALFWORLD_USER_TEMPLATE",
    "BAREBONE_BCB_SYSTEM_PROMPT",
    "BAREBONE_LLB_SYSTEM_PROMPT_BASE",
    "BAREBONE_LLB_USER_TEMPLATE",
]
