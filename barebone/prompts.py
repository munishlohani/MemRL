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


__all__ = ["BAREBONE_ALFWORLD_SYSTEM_PROMPT", "BAREBONE_ALFWORLD_USER_TEMPLATE"]
