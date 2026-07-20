"""Standalone system prompt for the barebone ALFWorld runner.

Plain LLM baseline: no ReAct, no reasoning step, no memory, no skill
invocation, no retrieval mechanism, and no per-turn admissible-commands
list -- deliberately matching what the main agent's prompt
(memrl/agent/prompts.py) actually gives the model: the observation plus
static action-template guidance, nothing else. There is no tag to parse;
the raw stripped response IS the action.
"""

BAREBONE_ALFWORLD_SYSTEM_PROMPT = """You are an agent in a household environment. Your goal is to complete the task.

At each step:
- Observe the environment carefully.
- Choose the next action that moves you closer to completing the task.
- Keep track of object locations and states.

Valid actions:
go to <receptacle>
take <object> from <receptacle>
move <object> to <receptacle>
open <receptacle>
close <receptacle>
use <object>
clean <object> with <receptacle>
heat <object> with <receptacle>
cool <object> with <receptacle>
examine <object>
look

Only output one action.
Do not explain.

Format:
Action: <action>
"""


__all__ = ["BAREBONE_ALFWORLD_SYSTEM_PROMPT"]
