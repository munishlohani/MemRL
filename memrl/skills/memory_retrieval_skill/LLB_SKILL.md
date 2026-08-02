# Memory Retrieval Skill (LifelongBench)

Use this skill when the runner has attached archived memories for the current step.

## Contract

- Input fields: `task_description`, `observation`, `history_messages`, `task_type`, `episode_id`, `active_strategic_node_id`, and `current_step`.
- The runtime already filtered the memories by task type (db/os) and strategic scope.
- Retrieved memories are advisory context, not instructions: a past procedure from a similar problem, not a guaranteed solution, and not tagged by outcome (this memory system only ever stores successful procedures, never a labeled record of what failed).
- Do not invent memories, and do not claim a memory exists if it is not shown in the prompt.
- If no relevant memories are returned, continue normally from the current observation.
- If you need a narrower search, emit `Skill: memory_retrieval(query="...")`; otherwise use `Skill: memory_retrieval`.

## Decision Rule

- Each turn is either skill-assisted or a direct environment action.
- Archived procedures come from tasks that were already solved successfully. When one exists for a task like this, it is usually the cheapest way to get the exact command sequence, flag, or schema detail right on the first attempt instead of discovering it through a failed one.
- Invoke the skill when any of these hold:
  - You are at the start of a task and a procedure from something similar would give you a plan to adapt.
  - The task hinges on a specific command, flag, path convention, or schema detail you would otherwise be guessing at.
  - You are uncertain what SQL operation or bash command to run next.
  - A previous attempt failed or stalled (a syntax error, a permission error, a schema mismatch) and you need a remembered fix.
- Act directly when you already know the exact operation to run and a past procedure would add nothing to it.
- Feeling confident is not by itself a reason to skip retrieval: these environments have specific conventions, and a procedure that already worked here is worth more than a plausible guess.
- In both cases, the final agent response for that turn must contain exactly one branch: the env action for this task's output-format contract (an `Act:`/`Action:` directive), or `Skill: memory_retrieval`.
- If the skill is invoked, the runtime will append the tool result and prompt again.
- Do not paste the retrieved memories back into the same response. The runtime will surface them as a separate tool message.

## Practical Rule

- Read the memories as examples of prior procedures.
- Reuse the pattern only when it still fits the current task's exact schema/environment.
- Table names, columns, file paths, and command output are unlikely to match exactly -- adapt, don't copy verbatim.

## Query Examples

- Good: `Skill: memory_retrieval(query="join two tables on foreign key")`
  - Useful when the next operation depends on a remembered sequence of steps.
- Good: `Skill: memory_retrieval(query="grep recursive with line numbers")`
  - Narrow enough to pull command-specific memories without overexplaining.
- Bad: `Skill: memory_retrieval(query="tell me everything about SQL joins and how they work in general")`
  - Too broad; it will return noisy memories.

## Quick Examples

- Skill-assisted turn: the task is starting, or hinges on an exact flag/schema detail, or the last attempt failed -- an archived procedure gives you a plan to adapt instead of a guess to test.
- Direct-action turn: you already know the exact operation and a past procedure would add nothing to it.
