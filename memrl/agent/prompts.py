# memp/agent/prompts.py

# This part is static during an episode. No hard-coded action-template list:
# the actual valid commands for this turn are given live by the environment
# as admissible_commands -- the agent copies one of those verbatim, rather
# than guessing from a static description of the action space.
SYSTEM_PROMPT = """You are controlling a text-based ALFWorld environment. Choose the NEXT action as ONE admissible command string. Output only the command, copied verbatim from the admissible list."""


# This template is for the user's message when memories are found.
WITH_MEMORY_PROMPT = """**Primary Goal:**
{task_description}

**Archived Memories (from your own past experiences):**
{retrieved_memories}

**Current Task Progress (recent steps):**
{history}
"""

# This template is for the user's message when no memories are found.
ZERO_SHOT_PROMPT = """**Primary Goal:**
{task_description}

**Archived Memories (from similar past tasks):**
No relevant memories were found. You must rely on your general knowledge.

**Current Task Progress (recent steps):**
{history}
"""
