"""Sleep consolidation package."""

from .clustering import (
    ClusteringStrategyBase,
    HDBSCANStrategy,
    KMeansClusteringStrategy,
    get_clustering_strategy,
)
from .prompts import (
    REVISE_STRATEGY_PROMPT,
    build_revise_strategy_prompt,
    format_cluster_contents,
)
from .service import SleepConsolidationService
from .checkpoint import SleepConsolidationCheckpoint
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
    "SleepConsolidationCheckpoint",
    "SleepConsolidationAction",
    "SleepConsolidationDecision",
    "SleepConsolidationResult",
    "StrategicScaffoldContext",
    "REVISE_STRATEGY_PROMPT",
    "build_revise_strategy_prompt",
    "format_cluster_contents",
]
