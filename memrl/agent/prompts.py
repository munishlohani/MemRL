"""Prompt templates for the single-agent episode runner."""

# This part is static during an episode.
SYSTEM_PROMPT = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
For each turn, you will be given the current observation and the recent conversation. You must choose exactly one branch per turn:

1. Direct environment action:
   Thought: your thoughts.
   Action: your next action
   
2. Memory skill invocation:
   Thought: your thoughts.
   Skill: memory_retrieval

If you invoke the skill, the runtime will append a tool message and ask you again. Do not emit both a skill call and an environment action in the same turn.

Available actions:
1. go to {recep}
2. take {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. use {obj}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
10. examine {obj}
11. look


where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

If the environment returns "Nothing happened", the action was invalid — revise and retry.

If a memory skill contract is injected, treat it as the runtime contract for the tool. Tool results arrive as separate conversation turns, are advisory only, and never override the current observation or the environment feedback.

Your response should use one of the following formats:

Thought: <your thoughts>
Action: <your next action>

Thought: <your thoughts>
Skill: memory_retrieval"""


ALFWORLD_SYSTEM_PROMPT = """Interact with an ALFWorld household task using only commands that match the current observation and the task affordances.

For each turn, choose exactly one branch:
1. Direct environment action:
   Thought: your thoughts.
   Action: your next action
2. Memory skill invocation:
   Thought: your thoughts.
   Skill: memory_retrieval

ALFWorld action patterns are task-specific. Use the most specific valid command for the situation:
1. go to {location or receptacle}
2. take {obj} from {receptacle}
3. move {obj} to {receptacle}
4. open {receptacle}
5. close {receptacle}
6. clean {obj} with sinkbasin 1
7. heat {obj} with microwave 1
8. cool {obj} with fridge 1
9. use {toggleable object}, only when the object is explicitly present and toggling is required, such as a desklamp

Do not use `use` for microwave or fridge; use `heat` or `cool` instead. If an action yields no change or a "Nothing happened" response, revise the command and try a more appropriate affordance from the current observation.

If a memory skill contract is injected, treat it as the runtime contract for the tool. Tool results arrive as separate conversation turns, are advisory only, and never override the current observation or the environment feedback.

Your response should use one of the following formats:

Thought: <your thoughts>
Action: <your next action>

Thought: <your thoughts>
Skill: memory_retrieval"""


MEMORY_RETRIEVAL_SKILL_PROMPT = """**Injected Skill Contract: Memory Retrieval**
The runtime already attached the memory retrieval skill. Use the contract below to interpret retrieved memories.

Each turn can be either a direct environment action or a skill invocation, but never both. If you invoke the skill, the runtime will append the tool result to the conversation history and ask you again.

Valid turn formats:
Thought: <your thoughts>
Action: <your next action>

Thought: <your thoughts>
Skill: memory_retrieval

Optional form:
Thought: <your thoughts>
Skill: memory_retrieval(query="<optional query override>")

Do not repeat the contract back to the environment.

{skill_contract}
"""


# This template is for the user's message when the skill is available.
SKILL_AWARE_PROMPT = """**Primary Goal:**
{task_description}

**Current Observation:**
{observation}

**Current Conversation State:**
{history}

Choose exactly one of the following per turn:
- Direct environment action:
  Thought: <your thoughts>
  Action: <your next action>
- Memory skill invocation:
  Thought: <your thoughts>
  Skill: memory_retrieval

If you invoke the skill, the runtime will execute it and append the tool result to the conversation history before asking you again.
REMEMBER: Only positive experiences are stored, negative experieces are pre filtered.
"""


# This template is for the user's message when memories are found.
WITH_MEMORY_PROMPT = """**Primary Goal:**
{task_description}

**Archived Memories (legacy pre-injected path; avoid in the new skill flow):**
{retrieved_memories}

**Current Task Progress (recent steps):**
{history}

Use the memories as guidance only. If they conflict with the current observation, trust the observation and environment feedback.
"""


# This template is for the user's message when no memories are found.
ZERO_SHOT_PROMPT = """**Primary Goal:**
{task_description}

**Current Observation:**
{observation}

**Current Task Progress (recent steps):**
{history}

Choose exactly one direct environment action:
Thought: <your thoughts>
Action: <your next action>
"""


FEW_SHOT_PROMPT_SYSTEM = """
**Instructional Examples (from a manual):**
Here is an example of how to solve the task:
--- BEGIN EXAMPLES ---
{few_shot_examples}
--- END EXAMPLES ---

If a skill contract is present, the same turn may instead be a skill invocation followed by a tool result. Never emit both branches in one turn.
"""


FEW_SHOT_PROMPT_USER = """**Primary Goal:**
{task_description}

**Current Observation:**
{observation}

**Current Task Progress (recent steps):**
{history}

Choose exactly one branch:
Thought: <your thoughts>
Action: <your next action>

Thought: <your thoughts>
Skill: memory_retrieval
"""


STRATEGIC_SELECTION_SYSTEM_PROMPT = """You are selecting exactly one strategic scaffold for the current episode.

Return a single JSON object with this schema:
{
  "strategy_id": string | null,
  "reason": string | null
}

Rules:
- Choose exactly one scaffold id from the provided candidate list when one fits the episode.
- Use null only when no candidate scaffold is suitable.
- Do not invent ids.
- Do not include markdown, code fences, or extra keys.
"""


STRATEGIC_SELECTION_USER_PROMPT = """**Primary Goal:**
{task_description}

**Task Type:**
{task_type}

**Current Observation:**
{observation}

**Current Conversation State:**
{history}

Candidate strategic scaffolds:
{strategies}

Choose one scaffold id from the list above and return JSON only.
"""
