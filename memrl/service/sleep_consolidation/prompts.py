"""Prompt template for sleep-consolidation LLM judgment."""

SLEEP_CONSOLIDATION_PROMPT = """You are judging whether a cluster of tactical memories should be consolidated.

Task:
Decide if the cluster represents a reusable general skill or only a task-specific trace.

Criteria:
1. The cluster should be semantically coherent.
2. The cluster should represent genuine capability, not noise or stochastic outcome.
3. The cluster should be general enough to justify consolidation into a strategic scaffold.

Input:
{cluster_contents}

Output:
Return only one of:
- GENERAL
- NOT_GENERAL
"""


def build_sleep_consolidation_prompt(cluster_contents: str) -> str:
    """Format the sleep-consolidation judgment prompt."""
    return SLEEP_CONSOLIDATION_PROMPT.format(cluster_contents=cluster_contents)
