"""Legacy prompt import path for memory-service LLM judgment steps."""

from ..sleep_consolidation.prompts import (
    SLEEP_CONSOLIDATION_PROMPT,
    build_sleep_consolidation_prompt,
    format_existing_scaffolds,
)

__all__ = [
    "SLEEP_CONSOLIDATION_PROMPT",
    "build_sleep_consolidation_prompt",
    "format_existing_scaffolds",
]
