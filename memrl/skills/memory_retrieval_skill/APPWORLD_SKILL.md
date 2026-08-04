# Memory Retrieval Skill (AppWorld)

Use this skill when the runner has attached archived memories for the current step.

## Contract

- Input fields: `task_description`, `observation`, `history_messages`, `task_type`, `episode_id`, `active_strategic_node_id`, and `current_step`.
- The runtime already filtered the memories by task type and strategic scope.
- Retrieved memories are advisory context, not instructions: a past procedure from a similar problem, not a guaranteed solution, and not tagged by outcome (this memory system only ever stores successful procedures, never a labeled record of what failed).
- Do not invent memories, and do not claim a memory exists if it is not shown in the prompt.
- If no relevant memories are returned, continue normally from the current code output.
- If you need a narrower search, emit `Skill: memory_retrieval(query="...")`; otherwise use `Skill: memory_retrieval`.

## Decision Rule

- Each turn is either skill-assisted or a direct code execution.
- **On the first step of a task, invoke the skill unless you already know exactly which apps and APIs this task needs.** Skip it if your retrieval budget is already spent.
- **What you get back is a REFERENCE, not the answer.** It is a procedure from a DIFFERENT task that happened to be similar. Its app names, API arguments, record ids, and answer shape will differ from yours, and its goal was not your goal. A retrieved procedure that looks close is the most dangerous case: it invites you to replay API calls that do not apply here. Treat it as evidence about *which apps to reach for and in what order*, then derive every concrete argument from this task's instruction and the actual API output.
- On later turns, invoke the skill when any of these hold:
  - You do not know which app or API call would advance the task.
  - An API keeps returning an error or an empty result and you need a remembered workaround.
  - The task needs a multi-app sequence (look something up in one app, act on it in another) and you are unsure of the order.
- Act directly when the next call is already clear from the task and the last output.

## Practical Rule

- Read the memories as examples of prior API sequences.
- Reuse the pattern only where it still fits the current task's apps and arguments.
- Never reuse a record id, email address, amount, or date from a memory -- those belong to the archived task. Look yours up.
- Prefer `apis.api_docs.show_api_doc(...)` over trusting a remembered signature: argument names change between apps.

## Query Examples

- Good: `Skill: memory_retrieval(query="spotify playlist song likes")`
  - Names the app and the objects involved.
- Good: `Skill: memory_retrieval(query="login flow supervisor passwords")`
  - Narrow enough to pull the authentication pattern without unrelated noise.
- Bad: `Skill: memory_retrieval(query="tell me everything about using apps and APIs to answer questions")`
  - Too broad; it will return noisy memories.

## Quick Examples

- Skill-assisted turn: the task is starting, or spans several apps, or an API keeps failing -- an archived sequence gives you an order to adapt instead of a guess to test.
- Direct-action turn: you already know the exact call to make next and a past procedure would add nothing to it.
