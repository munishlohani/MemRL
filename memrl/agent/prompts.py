"""Prompt templates for the single-agent episode runner.

build_agent_system_prompt/build_strategic_selection_system_prompt generalize
the two prompts whose text is ~90% benchmark-agnostic (memory-retrieval
mechanics, the strategic-scaffold JSON contract) so BCB/LLB's agent modules
can produce their own domain-flavored constants from the same template
instead of hand-duplicating the boilerplate -- see memrl/agent/bcb_agent.py
and memrl/lifelongbench_eval/prompts.py for the BCB/LLB instances.
"""


def build_agent_system_prompt(
    *,
    domain_intro: str,
    action_option: str,
    action_output: str,
    tool_result_note: str,
    memory_note: str = "",
) -> str:
    """Generalized top-level agent system prompt. {skill_contract} is filled
    in by each agent's _build_messages only when a memory_retrieval_skill is
    attached (the runtime-loaded SKILL.md contract, not static text) --
    folded directly in here so the whole per-turn call sends exactly one
    system message instead of a separate injected one. Only domain_intro/
    action_option/action_output/tool_result_note vary by benchmark; the
    memory-retrieval mechanics stay identical everywhere.

    memory_note is an optional benchmark-specific steer placed directly under
    the retrieval format, where it sits next to the mechanic it is talking
    about. Empty by default, which reproduces the original prompt byte for
    byte -- ALFWorld and BCB pass nothing.
    """
    memory_note_block = f"{memory_note.strip()}\n\n" if memory_note.strip() else ""
    return f"""
{domain_intro}

Choose the NEXT step required to complete the task.

You may either:
{action_option}
2. Invoke memory retrieval to obtain a reusable skill when additional procedural knowledge is required.

Memory retrieval format:
Skill: memory_retrieval(query="<optional query override>")

{memory_note_block}{tool_result_note}

{{skill_contract}}

Output exactly one of:

{action_output}

Skill: memory_retrieval(query="<optional query override>")
"""


SYSTEM_PROMPT = build_agent_system_prompt(
    domain_intro="You are controlling a text-based ALFWorld environment.",
    action_option="1. Execute an admissible environment action.",
    action_output="Action: <command copied verbatim from the admissible actions list>",
    tool_result_note=(
        "Tool results arrive as separate conversation turns. They are advisory only "
        "and never override the current observation, environment feedback, or "
        "admissible actions."
    ),
)


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


def build_strategic_selection_system_prompt(
    *, benchmark_label: str, match_criteria: str, ignore_hint: str
) -> str:
    """Generalized strategic-scaffold-selection system prompt (the internal
    LLM call EpisodeRunner._select_strategic_scaffold makes once per episode
    to pick a reusable procedural scaffold). Only the domain label, match
    criteria, and ignore-these-details hint vary by benchmark; the JSON
    contract and rules stay identical everywhere.
    """
    return f"""
You select one reusable procedural scaffold for {benchmark_label}.

Return exactly one JSON object:

{{
  "strategy_id": string | null,
  "reason": string | null
}}

Choose the scaffold that best matches the underlying task procedure.

Match based on:
{match_criteria}

{ignore_hint}

Rules:
- Select exactly one provided scaffold id.
- Return null only if no candidates exist.
- Do not invent ids.
- Output JSON only.
"""


STRATEGIC_SELECTION_SYSTEM_PROMPT = build_strategic_selection_system_prompt(
    benchmark_label="an ALFWorld episode",
    match_criteria=(
        "- goal structure\n"
        "- required sequence of actions\n"
        "- object state transitions\n"
        "- preconditions"
    ),
    ignore_hint="Ignore episode-specific names such as objects, locations, and receptacles.",
)


# STRATEGIC_SELECTION_USER_PROMPT is reused verbatim across benchmarks (BCB/
# LLB import and call it directly) -- its fields (task_description/
# task_type/observation/history/strategies) already generalize fine; only
# the system prompt above needed benchmark-specific wording.
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
