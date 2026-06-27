"""Sleep consolidation package.

This package will host clustering, prompt, and orchestration code for the
LLM-driven sleep consolidation pipeline.
"""

from .prompts import SLEEP_CONSOLIDATION_PROMPT, build_sleep_consolidation_prompt

__all__ = [
    "SLEEP_CONSOLIDATION_PROMPT",
    "build_sleep_consolidation_prompt",
]
