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
- "spawn": create a new d=1 strategic scaffold. Set summary to a concise reusable scaffold summary. Set target_scaffold_id to null.
- "absorb": use one existing d=1 scaffold. Set target_scaffold_id to the chosen scaffold id. Set summary to null.
- "discard": leave the tactical cluster as-is. Do not create a new scaffold, do not reparent the cluster, and do not otherwise modify graph structure. Set summary and target_scaffold_id to null.
- The summary field becomes SkillRepresentation.content.
- Do not output embeddings, Q-values, explanations, markdown, or extra keys.
- If there are no suitable existing scaffolds, choose spawn or discard, not absorb.

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
