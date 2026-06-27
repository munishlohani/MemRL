"""Sleep consolidation package.

This package will host clustering, prompt, and orchestration code for the
LLM-driven sleep consolidation pipeline.
"""

from .clustering import (
    ClusteringStrategyBase,
    HDBSCANStrategy,
    KMeansClusteringStrategy,
    get_clustering_strategy,
)
from .prompts import (
    SLEEP_CONSOLIDATION_PROMPT,
    build_sleep_consolidation_prompt,
    format_existing_scaffolds,
    format_cluster_contents,
)
from .service import SleepConsolidationService
from .types import (
    SleepConsolidationAction,
    SleepConsolidationDecision,
    SleepConsolidationResult,
    StrategicScaffoldContext,
)

__all__ = [
    "ClusteringStrategyBase",
    "KMeansClusteringStrategy",
    "HDBSCANStrategy",
    "get_clustering_strategy",
    "SleepConsolidationService",
    "SleepConsolidationAction",
    "SleepConsolidationDecision",
    "SleepConsolidationResult",
    "StrategicScaffoldContext",
    "SLEEP_CONSOLIDATION_PROMPT",
    "build_sleep_consolidation_prompt",
    "format_existing_scaffolds",
    "format_cluster_contents",
]
