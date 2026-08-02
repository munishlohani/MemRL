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
- **Only invoke the memory retrieval skill when it is actually necessary.** Retrieval is not free: an irrelevant or misleading archived memory can bias you toward the wrong query or command even on a task you were already capable of solving directly.
- Invoke the skill only when at least one of these is true:
  - You are genuinely uncertain what SQL operation or bash command to run next.
  - You have seen this kind of failure or dead end before (a syntax error, a permission error, a schema mismatch) and don't know how to get past it.
  - The task has just failed or stalled and you need a remembered fix.
- Skip the skill and act directly whenever the next operation is already clear from the current observation and task progress -- this is the common case for straightforward tasks. Do not retrieve "just in case" or out of habit.
- In both cases, the final agent response for that turn must contain exactly one branch: the env action for this task's STRICT OUTPUT FORMAT (an `Act:`/`Action:` directive), or `Skill: memory_retrieval`.
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

- Skill-assisted turn: the current observation is ambiguous or the last attempt failed, so archived memories are useful for choosing the next step.
- Direct-action turn: the current observation already makes the next operation obvious, so the agent should act without relying on memory.
