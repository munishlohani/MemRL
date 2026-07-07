"""Prompt templates for sleep-consolidation LLM decisions."""

from __future__ import annotations

from typing import Sequence

from .types import StrategicScaffoldContext

SLEEP_CONSOLIDATION_PROMPT = """You are deciding how to consolidate one tactical cluster into the memory graph.

Return a single JSON object and nothing else.

Schema:
{{
  "action": "spawn" | "absorb" | "discard",
  "summary": string | null,
  "target_scaffold_id": string | null
}}

Rules:
- "spawn": create a new d=1 strategic scaffold. Set summary per the Summary Spec below. Set target_scaffold_id to null.
- "absorb": use one existing d=1 scaffold. Set target_scaffold_id to the chosen scaffold id. Set summary to null.
- "discard": leave the tactical cluster as-is. Do not create a new scaffold, do not reparent the cluster, and do not otherwise modify graph structure. Set summary and target_scaffold_id to null.
- The summary field becomes SkillRepresentation.content.
- Do not output embeddings, Q-values, explanations, markdown, or extra keys.
- If there are no suitable existing scaffolds, choose spawn or discard, not absorb.

Summary Spec (applies only when action is "spawn" -- a strategic scaffold is
a reusable STRATEGY, not a record of one task):
- Object abstraction, procedure retention. Replace specific object/receptacle/appliance names (tomato, mug, apple, sink, microwave, fridge) with role descriptors (the target object, the destination receptacle, the tool, the appliance). Do NOT abstract the procedure -- the action sequence, locations, and ordering constraints stay concrete and literal.
- Precondition-guarded steps. Where cluster members below share a precondition or failure-prone step (must be holding the object; must be at the correct receptacle; appliance must be open before use), attach it to that step as an explicit checkable condition, and instruct skipping any step whose postcondition already holds.
- Structural, not descriptive, generality. Generalize over WHAT is manipulated, never over HOW you do it. A summary that would fit any task ("complete the objective efficiently") is a failure; a summary naming one concrete object ("cool the tomato") is also a failure. The target sits between those two: object-agnostic, procedure-specific.
- Grounding in shared evidence. State only procedure elements present across MULTIPLE cluster members below. A step that appears in only one member is an instance detail -- omit it.
- Shape: one short title line, then 3-6 ordered imperative steps, each carrying its precondition in parentheses.

Summary Spec examples (for calibration only -- derive the real summary from the cluster below, do not copy these):

Example cluster: [1] "goal: cool the tomato and place it in the microwave; steps: pick up the tomato, open the fridge, put the tomato in the fridge, wait, take the tomato out, open the microwave, put the tomato in the microwave" [2] "goal: cool the potato and place it on the countertop; steps: pick up the potato, open the fridge, put the potato in the fridge, wait, take the potato out, put the potato on the countertop"
Good summary (object-agnostic, procedure-specific, precondition-guarded):
"Cool an object and place it at a destination.
1. Pick up the target object (must not already be holding it).
2. Open the fridge (skip if already open).
3. Put the object inside and wait for it to cool.
4. Take the cooled object back out.
5. Go to the destination receptacle, opening it first if it is a closed appliance.
6. Place the object at the destination."
Bad summary (too specific, rejected): "Cool the tomato in the fridge and put it in the microwave."
Bad summary (too vague, rejected): "Complete the cooling task efficiently."

Example cluster: [1] "goal: clean the mug and put it on the shelf" [2] "goal: clean the plate and put it in the cabinet"
Good summary:
"Clean an object and place it at a destination.
1. Pick up the target object (must not already be holding it).
2. Go to the sink (must be holding the target object).
3. Turn on the faucet, wash the object, then turn the faucet off.
4. Go to the destination receptacle, opening it first if it is a closed container.
5. Place the cleaned object at the destination."

Cluster:
{cluster_contents}

Existing d=1 scaffolds:
{existing_scaffolds}
"""


def format_cluster_contents(cluster_texts: Sequence[str]) -> str:
    """Format cluster texts into a compact prompt-ready block."""
    if not cluster_texts:
        return "(empty cluster)"
    return "\n".join(
        f"{idx + 1}. {text.strip()}"
        for idx, text in enumerate(cluster_texts)
    )


def format_existing_scaffolds(
    existing_scaffolds: Sequence[StrategicScaffoldContext],
) -> str:
    """Format existing strategic scaffolds for prompt input."""
    if not existing_scaffolds:
        return "(none)"
    return "\n".join(
        f"- {scaffold.node_id}: {scaffold.summary.strip()}"
        for scaffold in existing_scaffolds
    )


def build_sleep_consolidation_prompt(
    cluster_contents: str,
    existing_scaffolds: str,
) -> str:
    """Format the structured sleep-consolidation prompt."""
    return SLEEP_CONSOLIDATION_PROMPT.format(
        cluster_contents=cluster_contents,
        existing_scaffolds=existing_scaffolds,
    )
