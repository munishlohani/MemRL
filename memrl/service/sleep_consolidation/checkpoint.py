"""Sleep consolidation trigger checkpoint."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..strategies import ClusterStrategy
from .service import SleepConsolidationService
from ...utils.event_logging import log_event

logger = logging.getLogger(__name__)


class SleepConsolidationCheckpoint:
    """
    Encapsulates the sleep consolidation trigger and delegates execution to
    MemoryService.sleep_consolidate().

    Runs after all episode finalizations in a batch are complete.
    Checks if consolidation should fire based on unconsolidated tactical count.
    """

    def __init__(
        self,
        memory_service: Any,
        llm_provider: Optional[Any] = None,
        memory_config: Optional[Any] = None,
    ) -> None:
        self.memory_service = memory_service
        self.llm_provider = llm_provider
        self.memory_config = memory_config
        self.graph = memory_service.graph

    def check_and_trigger(self) -> Optional[Dict[str, Any]]:
        """Check whether sleep consolidation should fire."""
        n_sleep = getattr(self.memory_config, "n_sleep", None)
        if n_sleep is None:
            logger.debug("Sleep consolidation disabled: n_sleep is unset")
            return None

        unconsolidated_count = sum(
            1
            for node in self.graph.nodes.values()
            if node.is_tactical and not node.consolidated
        )

        n_sleep = int(n_sleep)
        if unconsolidated_count < n_sleep:
            logger.debug(
                "Sleep consolidation not triggered: "
                "unconsolidated_count=%s < n_sleep=%s",
                unconsolidated_count,
                n_sleep,
            )
            return None

        logger.info(
            "Sleep consolidation triggered: unconsolidated=%s, n_sleep=%s",
            unconsolidated_count,
            n_sleep,
        )
        log_event(
            logger,
            "sleep_consolidation.checkpoint_trigger",
            unconsolidated_count=unconsolidated_count,
            n_sleep=n_sleep,
            current_step=self.graph.current_step,
        )
        return self._run_consolidation(
            theta_consolidate=self._resolved_theta_consolidate(),
            unconsolidated_count=unconsolidated_count,
        )

    def _resolved_theta_consolidate(self) -> float:
        threshold = getattr(self.memory_config, "theta_consolidate", None)
        return float(threshold) if threshold is not None else 0.0

    def _run_consolidation(
        self, *, theta_consolidate: float, unconsolidated_count: int
    ) -> Dict[str, Any]:
        """Execute the consolidation pipeline through MemoryService."""
        if self.llm_provider is None:
            raise ValueError("llm_provider is required for sleep consolidation")

        cluster_strategy = self._resolved_cluster_strategy()
        sleep_service = SleepConsolidationService(
            llm_provider=self.llm_provider,
            cluster_strategy=cluster_strategy,
        )
        stats: Dict[str, Any] = {}
        results = self.memory_service.sleep_consolidate(
            sleep_service,
            theta_consolidate=theta_consolidate,
            stats_out=stats,
        )
        summary = {
            "consolidation_ran": True,
            "timestamp": self.graph.current_step,
            "trigger_step": self.graph.current_step,
            "unconsolidated_count": unconsolidated_count,
            "num_results": len(results),
            "actions": [result.action.value for result in results],
            "eligible_count": stats.get("eligible_count"),
            "cluster_count": stats.get("cluster_count"),
            "cluster_sizes": stats.get("cluster_sizes"),
            "cluster_davies_bouldin": stats.get("cluster_davies_bouldin"),
            "action_counts": stats.get("action_counts"),
        }
        logger.info(
            "Sleep consolidation complete: timestamp=%s, num_results=%s",
            summary["timestamp"],
            summary["num_results"],
        )
        log_event(
            logger,
            "sleep_consolidation.checkpoint_done",
            timestamp=summary["timestamp"],
            num_results=summary["num_results"],
            actions=summary["actions"],
        )
        return summary

    def _resolved_cluster_strategy(self) -> ClusterStrategy:
        if self.memory_config is None:
            return ClusterStrategy.KMEANS

        getter = getattr(self.memory_config, "get_cluster_strategy", None)
        if callable(getter):
            return getter()

        strategy_name = str(getattr(self.memory_config, "cluster_strategy", "kmeans"))
        return ClusterStrategy.from_string(strategy_name)
