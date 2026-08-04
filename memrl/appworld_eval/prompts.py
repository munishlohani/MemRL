"""AppWorld prompt construction.

Mirrors ALFWorld's four-piece prompt architecture (memrl/agent/prompts.py) via
the same builders BCB and LLB use, so all four benchmarks share one mechanism
and differ only in domain wording:

  1. APPWORLD_SYSTEM_PROMPT      -- via build_agent_system_prompt, carries the
                                    {skill_contract} slot filled from
                                    APPWORLD_SKILL.md
  2. APPWORLD_SKILL_AWARE_PROMPT -- user turn when a retrieval skill is attached
  3. APPWORLD_ZERO_SHOT_PROMPT   -- user turn when it is not
  4. APPWORLD_STRATEGIC_SELECTION_SYSTEM_PROMPT -- via
                                    build_strategic_selection_system_prompt

AppWorld's interaction model sits between BCB and LLB: the action is a Python
code block (like BCB) but the episode is multi-turn with a running history and
the stdout of the previous block (like LLB). Execution is STATEFUL across turns
-- variables persist like a notebook -- which the prompt has to say explicitly,
or the model re-imports and re-logs-in every turn and burns its step budget.
"""

from __future__ import annotations

from memrl.agent.prompts import (
    build_agent_system_prompt,
    build_strategic_selection_system_prompt,
)


APPWORLD_SYSTEM_PROMPT = (
    build_agent_system_prompt(
        domain_intro=(
            "You are an AI assistant completing a task for your supervisor by calling app "
            "APIs (Spotify, Gmail, Venmo, Amazon, phone, file system, and others) from "
            "Python code."
        ),
        action_option="1. Execute a block of Python code against the app APIs.",
        action_output="Action:\n```python\n<python code to execute this turn>\n```",
        tool_result_note=(
            "Tool results arrive as separate conversation turns. They are advisory only "
            "and never override the current code output, an API error message, or the "
            "task instruction."
        ),
    ).strip()
    + "\n\n"
    "How the environment works:\n"
    "- Code execution is STATEFUL across turns, like a notebook. Variables, imports, and "
    "logins you set up in one turn are still available in the next -- do not repeat setup "
    "work you have already done.\n"
    "- `apis` is already available. You do not import it.\n"
    "- `print(...)` anything you need to see: only stdout comes back to you.\n"
    "- Discover an app before guessing at it: `print(apis.api_docs.show_app_descriptions())`, "
    "then `print(apis.api_docs.show_api_descriptions(app_name='...'))`, then "
    "`print(apis.api_docs.show_api_doc(app_name='...', api_name='...'))`.\n"
    "- Credentials come from the supervisor app, not from you: "
    "`print(apis.supervisor.show_account_passwords())`.\n"
    "- When the task asks a question, submit the answer with "
    "`apis.supervisor.complete_task(answer=...)`. When it asks for actions only, call "
    "`apis.supervisor.complete_task()`. The episode is not scored as complete until you "
    "call it.\n"
    "- Work in small steps. One focused block per turn beats a long script, because you "
    "see the output and can correct course.\n"
)


APPWORLD_SKILL_AWARE_PROMPT = """**Primary Goal:**
Task:
{task_description}

Retrieved skill information is available only after invoking memory retrieval.

Interaction history:
{history}

Output of your last code block:
{observation}

Choose the next step.

If acting:
Action:
```python
<python code to execute this turn>
```

If using memory:
Skill: memory_retrieval(query="<optional query override>")

On the first step, use memory retrieval unless you already know exactly which APIs this
task needs or the skill budget has been exhausted.

Anything already retrieved is a reference from a similar-but-different task, not a script
to replay. The app names, record ids, and argument values will differ. Adapt it, and prefer
the actual code output wherever the two disagree.
"""


APPWORLD_ZERO_SHOT_PROMPT = """Task:
{task_description}

Interaction history:
{history}

Output of your last code block:
{observation}
"""


APPWORLD_STRATEGIC_SELECTION_SYSTEM_PROMPT = build_strategic_selection_system_prompt(
    benchmark_label="an AppWorld app-API task",
    match_criteria=(
        "- goal structure (what the supervisor is ultimately asking for)\n"
        "- which apps are involved and in what order\n"
        "- the discovery/authentication sequence the task needs\n"
        "- how results are filtered, aggregated, or paginated before answering"
    ),
    ignore_hint=(
        "Ignore episode-specific values such as person names, record ids, playlist or "
        "email titles, amounts, and dates."
    ),
)


__all__ = [
    "APPWORLD_SYSTEM_PROMPT",
    "APPWORLD_SKILL_AWARE_PROMPT",
    "APPWORLD_ZERO_SHOT_PROMPT",
    "APPWORLD_STRATEGIC_SELECTION_SYSTEM_PROMPT",
]
