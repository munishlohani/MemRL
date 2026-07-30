"""End-of-training summary.json: per-epoch aggregates plus where this run's
outputs live. Written once the training loop finishes (or is interrupted --
whatever epochs completed by then are still included), so a single file
answers "how did this run go, epoch by epoch" without grepping logs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class EpochStatsAccumulator:
    """Accumulates per-section summaries (each EpisodeRunner.run() call)
    into one row for the epoch they belong to.

    success_rate/mean_reward/mean_steps are weighted by episode count, not
    section count, since the last section of an epoch can legitimately
    have fewer counted episodes than batch_size (duplicate-slot exclusion
    when num_games isn't evenly divisible by batch_size).
    """

    def __init__(self) -> None:
        self.episode_count = 0
        self.reward_sum = 0.0
        self.steps_sum = 0.0
        self.success_sum = 0.0
        self.cluster_count = 0

    def add_section(self, summary: Dict[str, Any]) -> None:
        counted = len(summary.get("episodes") or [])
        self.episode_count += counted
        self.reward_sum += float(summary.get("mean_reward", 0.0)) * counted
        self.steps_sum += float(summary.get("mean_steps", 0.0)) * counted
        self.success_sum += float(summary.get("success_rate", 0.0)) * counted
        # Sleep consolidation doesn't fire every section -- only count
        # clusters actually formed during sections where it did.
        sleep_summary = summary.get("sleep_consolidation")
        if sleep_summary:
            self.cluster_count += int(sleep_summary.get("cluster_count") or 0)

    def finalize(self, epoch: int, runner: Any) -> Dict[str, Any]:
        """strategic_nodes/tactical_nodes are a point-in-time snapshot of
        the skill graph at the end of this epoch, not "created this
        epoch" -- read straight off runner.memory_service.graph."""
        graph = getattr(runner.memory_service, "graph", None)
        strategic_nodes = len(graph.nodes_at_depth(1)) if graph is not None else 0
        tactical_nodes = len(graph.nodes_at_depth(2)) if graph is not None else 0
        return {
            "epoch": epoch,
            "episodes": self.episode_count,
            "success_rate": (self.success_sum / self.episode_count) if self.episode_count else 0.0,
            "mean_reward": (self.reward_sum / self.episode_count) if self.episode_count else 0.0,
            "mean_steps": (self.steps_sum / self.episode_count) if self.episode_count else 0.0,
            "strategic_nodes": strategic_nodes,
            "tactical_nodes": tactical_nodes,
            "clusters": self.cluster_count,
        }


def write_training_summary(run_dir: Path, epoch_summaries: List[Dict[str, Any]]) -> Path:
    payload = {
        "output_dir": str(run_dir),
        "epochs": epoch_summaries,
    }
    summary_path = Path(run_dir) / "summary.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Wrote training summary to %s", summary_path)
    except Exception:
        logger.warning("Failed to write training summary to %s", summary_path, exc_info=True)
    return summary_path


__all__ = ["EpochStatsAccumulator", "write_training_summary"]
