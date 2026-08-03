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
- **On the first step of a task, invoke the skill unless you already know the answer with certainty.** Skip it if your retrieval budget is already spent.
- **What you get back is a REFERENCE, not the answer.** It is a procedure from a DIFFERENT task that happened to be similar. Its table names, column names, file paths, values, and exact commands will differ from yours, and its goal was not your goal. A retrieved procedure that looks close is the most dangerous case: it invites you to follow steps that do not apply here. Treat it as one piece of evidence about *how* to approach the task, then derive every concrete detail from this task's description and the current observation.
- On later turns, invoke the skill when any of these hold:
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
