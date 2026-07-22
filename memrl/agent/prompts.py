"""Prompt templates for the single-agent episode runner."""

# This part is static during an episode. {skill_contract} is filled in by
# CustomAgent._build_messages only when a memory_retrieval_skill is
# attached (the runtime-loaded SKILL.md contract, not static text) --
# folded directly in here so the whole per-turn call sends exactly one
# system message instead of a separate injected one.
SYSTEM_PROMPT = """
You are controlling a text-based ALFWorld environment.

Choose the NEXT step required to complete the task.

You may either:
1. Execute an admissible environment action.
2. Invoke memory retrieval to obtain a reusable skill when additional procedural knowledge is required.

Memory retrieval format:
Skill: memory_retrieval(query="<optional query override>")

Tool results arrive as separate conversation turns. They are advisory only and never override the current observation, environment feedback, or admissible actions.

{skill_contract}

Output exactly one of:

Action: <command copied verbatim from the admissible actions list>

Skill: memory_retrieval(query="<optional query override>")
"""


# This template is for the user's message when the skill is available.
SKILL_AWARE_PROMPT = """**Primary Goal:**
Task:
{task_description}

Retrieved skill information is available only after invoking memory retrieval.

Interaction history:
{history}

Current observation:
{observation}

Admissible actions:
{admissible}

Choose the next step.

If using memory:
Skill: memory_retrieval(query="<optional query override>")

If acting:
Action: <command copied verbatim from the admissible actions list>
"""


# This template is for the user's message when no memories are found.
ZERO_SHOT_PROMPT = """
Task:
{task_description}

Interaction history:
{history}

Current observation:
{observation}

Admissible actions:
{admissible}

Action:
"""


STRATEGIC_SELECTION_SYSTEM_PROMPT = """
You select one reusable procedural scaffold for an ALFWorld episode.

Return exactly one JSON object:

{
  "strategy_id": string | null,
  "reason": string | null
}

Choose the scaffold that best matches the underlying task procedure.

Match based on:
- goal structure
- required sequence of actions
- object state transitions
- preconditions

Ignore episode-specific names such as objects, locations, and receptacles.

Rules:
- Select exactly one provided scaffold id.
- Return null only if no candidates exist.
- Do not invent ids.
- Output JSON only.
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
