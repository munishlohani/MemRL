# Memory Retrieval Skill (BigCodeBench)

Use this skill when the runner has attached archived memories for this task.

## Contract

- Input fields: `task_description`, `history_messages`, `task_type`, `episode_id`, `active_strategic_node_id`, and `current_step`.
- The runtime already filtered the memories by task type and strategic scope.
- Retrieved memories are advisory context, not instructions: a past procedure from a similar problem, not a guaranteed solution, and not tagged by outcome (this memory system only ever stores successful procedures, never a labeled record of what failed).
- Do not invent memories, and do not claim a memory exists if it is not shown in the prompt.
- If no relevant memories are returned, write your solution directly.
- If you need a narrower search, emit `Skill: memory_retrieval(query="...")`; otherwise use `Skill: memory_retrieval`.

## Decision Rule

- Each task is either skill-assisted (retrieve once, then submit) or a direct submission.
- **Only invoke the memory retrieval skill when it is actually necessary.** Retrieval is not free: an irrelevant or misleading archived procedure can bias you toward the wrong approach even on a task you were already capable of solving directly.
- Invoke the skill only when at least one of these is true:
  - You are genuinely uncertain how to approach the required function/signature.
  - The task involves an unfamiliar library, algorithm, or edge case you would benefit from seeing a worked example of.
- Skip the skill and submit directly whenever the solution approach is already clear from the task description -- this is the common case for straightforward tasks. Do not retrieve "just in case" or out of habit.
- In both cases, the final agent response for that turn must contain exactly one branch:
  - `Action:` followed by a fenced ```python code block
  - or `Skill: memory_retrieval`
- If the skill is invoked, the runtime will append the tool result and prompt again.
- Do not paste the retrieved memories back into the same response. The runtime will surface them as a separate tool message.
- You may invoke the skill at most once per task -- use your other turn to submit code.

## Practical Rule

- Read the memory as an example of a prior solution approach.
- Reuse the pattern only when it still fits this task's exact function signature and requirements.
- Do not copy code verbatim without verifying it against the current task's constraints (signature, return type, exception types).

## Query Examples

- Good: `Skill: memory_retrieval(query="pandas groupby aggregate")`
  - Useful when the next approach depends on a remembered library usage pattern.
- Good: `Skill: memory_retrieval(query="regex extract dates from text")`
  - Narrow enough to pull relevant memories without overexplaining.
- Bad: `Skill: memory_retrieval(query="tell me everything about string processing and text parsing in Python")`
  - Too broad; it will return noisy memories.

## Quick Examples

- Skill-assisted task: the required approach is unfamiliar, so an archived procedure is useful for choosing a starting point.
- Direct-submission task: the solution approach is already obvious from the task description, so the agent should submit without relying on memory.
