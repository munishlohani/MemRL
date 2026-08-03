"""Prompt template for sleep-consolidation Pass 2 (content authoring).

Pass 1 (structural: spawn/absorb/discard) is algorithmic -- cosine
similarity against a threshold, no LLM call, no prompt. This module only
covers Pass 2: the single LLM call per scaffold that authors or revises its
strategic `content`, whether triggered by a spawn, an absorb, or reflection
over accumulated failures with no structural change at all.
"""

from __future__ import annotations

from typing import Sequence

REVISE_STRATEGY_PROMPT = """You are authoring or revising the reusable strategic content for one d=1 scaffold -- the reasoning frame an agent is given before starting an episode of this kind. A scaffold holds a reusable STRATEGY, not a record of one task.

Return the strategy text only -- no JSON, no markdown, no explanations, nothing before or after it.

Rules:
- If CURRENT STRATEGY is non-empty, REVISE it. Preserve existing steps unless the evidence below directly contradicts them. Do not regenerate from scratch.
- Corrections derived from failures must be written PRESCRIPTIVELY in the numbered steps.
  BAD:  "Avoid heating before retrieving."
  GOOD: "Retrieve the target object before heating it."
- Separately from the steps, you MUST end the output with a `REFLECTION:` section naming what
  actually went wrong in the FAILED ATTEMPTS and what to watch for. This is the only place
  failures are visible as failures -- the steps above deliberately read as ordinary
  instructions, so without this section there is no record that anything was learned from a
  failure. Write `REFLECTION:` followed by 1-3 short lines. If FAILED ATTEMPTS is "(none)",
  write exactly `REFLECTION:` then `(none yet)`. Never invent a failure that is not in the
  evidence.
- FAILED ATTEMPTS are cases where this strategy was appropriate and still failed. They indicate flaws in the procedure, not in strategy selection.

Summary Spec (applies whether this is a fresh strategy or a revision):
- Object abstraction, procedure retention. Replace specific object/receptacle/appliance names (tomato, mug, apple, sink, microwave, fridge) with role descriptors (the target object, the destination receptacle, the tool, the appliance). Do NOT abstract the procedure -- the action sequence, locations, and ordering constraints stay concrete and literal.
- Precondition-guarded steps. Where the evidence below shares a precondition or failure-prone step (must be holding the object; must be at the correct receptacle; appliance must be open before use), attach it to that step as an explicit checkable condition, and instruct skipping any step whose postcondition already holds.
- Structural, not descriptive, generality. Generalize over WHAT is manipulated, never over HOW you do it. A strategy that would fit any task ("complete the objective efficiently") is a failure; a strategy naming one concrete object ("cool the tomato") is also a failure. The target sits between those two: object-agnostic, procedure-specific.
- Grounding in shared evidence. State only procedure elements present across MULTIPLE examples below. A step that appears in only one example is an instance detail -- omit it.
- Shape: one short title line, then 3-6 ordered imperative steps, each carrying its precondition in parentheses, then the `REFLECTION:` section. Keep it to 3-6 steps even across many revisions -- if new evidence adds a step, fold it into an existing one or drop a step that only ever appeared in a single example. A growing list of "If X is required" branches means the strategy is turning into a catch-all checklist, which is the same failure as being too vague.

Summary Spec examples (for calibration only -- derive the real strategy from the evidence below, do not copy these):

Example cluster: [1] "goal: cool the tomato and place it in the microwave; steps: pick up the tomato, open the fridge, put the tomato in the fridge, wait, take the tomato out, open the microwave, put the tomato in the microwave" [2] "goal: cool the potato and place it on the countertop; steps: pick up the potato, open the fridge, put the potato in the fridge, wait, take the potato out, put the potato on the countertop"
Good strategy (object-agnostic, procedure-specific, precondition-guarded):
"Cool an object and place it at a destination.
1. Pick up the target object (must not already be holding it).
2. Open the fridge (skip if already open).
3. Put the object inside and wait for it to cool.
4. Take the cooled object back out.
5. Go to the destination receptacle, opening it first if it is a closed appliance.
6. Place the object at the destination.

REFLECTION:
(none yet)"
Bad strategy (too specific, rejected): "Cool the tomato in the fridge and put it in the microwave."
Bad strategy (too vague, rejected): "Complete the cooling task efficiently."

Example with a failure present -- note the correction appears BOTH prescriptively in the steps AND named in REFLECTION:
Example cluster: [1] "goal: clean the mug and put it on the shelf" [2] "goal: clean the plate and put it in the cabinet"
Example failed attempt: [1] "went to the sink and turned on the faucet, but was not holding the plate, so nothing was washed"
Good strategy:
"Clean an object and place it at a destination.
1. Pick up the target object (must not already be holding it).
2. Go to the sink (must be holding the target object).
3. Turn on the faucet, wash the object, then turn the faucet off.
4. Go to the destination receptacle, opening it first if it is a closed container.
5. Place the cleaned object at the destination.

REFLECTION:
Washing was attempted without holding the object, so nothing happened. Confirm the object is in hand before reaching the sink."

CURRENT STRATEGY (empty if new):
{current_summary}

SUCCESSFUL EXAMPLES:
{positive_evidence}

FAILED ATTEMPTS (strategy was appropriate; it still failed):
{negative_evidence}
"""


def format_cluster_contents(texts: Sequence[str]) -> str:
    """Format a list of texts into a compact, numbered prompt-ready block.

    Reused for both `positive_evidence` (cluster member / absorbed-node
    contents) and `negative_evidence` (failure traces) in the Pass-2 prompt
    -- both are just numbered lists of prior text.
    """
    if not texts:
        return "(none)"
    return "\n".join(
        f"{idx + 1}. {text.strip()}"
        for idx, text in enumerate(texts)
    )


def build_revise_strategy_prompt(
    current_summary: str,
    positive_evidence: str,
    negative_evidence: str,
) -> str:
    """Format the Pass-2 content-authoring prompt.

    `positive_evidence`/`negative_evidence` are already-formatted text
    blocks (see `format_cluster_contents`), not raw lists.
    """
    return REVISE_STRATEGY_PROMPT.format(
        current_summary=current_summary.strip() or "(none -- this is a new scaffold)",
        positive_evidence=positive_evidence,
        negative_evidence=negative_evidence,
    )
