"""Sleep consolidation orchestration.

Two passes (Project.md "Reflection: Two-Pass Sleep Consolidation"):
  Pass 1 -- `decide_cluster_structure`: algorithmic, no LLM call. A cosine-
    similarity threshold against existing scaffold embeddings decides
    spawn/absorb/discard per cluster.
  Pass 2 -- `revise_strategy`: the sole LLM call, one per scaffold with
    changed evidence this sleep event (not per cluster -- `MemoryService.
    sleep_consolidate` groups by scaffold before calling this).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from ...providers.base import BaseLLM
from ..retrievers import cosine_similarity
from ..strategies import ClusterStrategy
from .clustering import ClusteringStrategyBase, compute_davies_bouldin_index, get_clustering_strategy, mean_embedding
from .prompts import build_revise_strategy_prompt, format_cluster_contents
from .types import (
    SleepConsolidationAction,
    SleepConsolidationDecision,
    SleepConsolidationResult,
    StrategicScaffoldContext,
)
from ...utils.event_logging import log_event


logger = logging.getLogger(__name__)


class SleepConsolidationService:
    """Clustering + Pass 1 structural decisions + Pass 2 content authoring."""

    def __init__(
        self,
        llm_provider: BaseLLM,
        clustering_strategy: Optional[ClusteringStrategyBase] = None,
        *,
        cluster_strategy: ClusterStrategy = ClusterStrategy.KMEANS,
        n_clusters: Optional[int] = None,
        random_state: int = 0,
        theta_absorb: float = 0.75,
        n_min_spawn: int = 3,
    ) -> None:
        self.llm_provider = llm_provider
        self.n_min_spawn = int(n_min_spawn)
        self.clustering_strategy = clustering_strategy or get_clustering_strategy(
            cluster_strategy,
            n_clusters=n_clusters,
            random_state=random_state,
            n_min_spawn=self.n_min_spawn,
        )
        self.theta_absorb = float(theta_absorb)

    def cluster_embeddings(
        self,
        embeddings: Sequence[Sequence[float]],
    ) -> List[List[int]]:
        """Group candidate embeddings into clusters."""
        return self.clustering_strategy.cluster(embeddings)

    def decide_cluster_structure(
        self,
        cluster_embeddings: Sequence[Sequence[float]],
        *,
        existing_scaffolds: Sequence[StrategicScaffoldContext] = (),
    ) -> SleepConsolidationDecision:
        """Pass 1: algorithmic spawn/absorb/discard decision for one cluster.

        Absorbs into whichever existing scaffold has the highest cosine
        similarity between this cluster's centroid and the scaffold's own
        embedding, if that similarity clears `theta_absorb`. Otherwise
        spawns a new scaffold if the cluster is large enough
        (`n_min_spawn`), else discards it.
        """
        cluster_size = len(cluster_embeddings)
        centroid = mean_embedding(cluster_embeddings)

        best_scaffold: Optional[StrategicScaffoldContext] = None
        best_similarity = float("-inf")
        for scaffold in existing_scaffolds:
            if not scaffold.embedding:
                continue
            similarity = cosine_similarity(centroid, scaffold.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_scaffold = scaffold

        if best_scaffold is not None and best_similarity > self.theta_absorb:
            decision = SleepConsolidationDecision(
                action=SleepConsolidationAction.ABSORB,
                target_scaffold_id=best_scaffold.node_id,
            )
        elif cluster_size >= self.n_min_spawn:
            decision = SleepConsolidationDecision(action=SleepConsolidationAction.SPAWN)
        else:
            decision = SleepConsolidationDecision(action=SleepConsolidationAction.DISCARD)

        log_event(
            logger,
            "sleep_consolidation.structure_decision",
            cluster_size=cluster_size,
            best_scaffold_id=best_scaffold.node_id if best_scaffold is not None else None,
            best_similarity=None if best_scaffold is None else best_similarity,
            theta_absorb=self.theta_absorb,
            n_min_spawn=self.n_min_spawn,
            action=decision.action.value,
        )
        return decision

    def revise_strategy(
        self,
        *,
        current_summary: str,
        positive_evidence: Sequence[str],
        negative_evidence: Sequence[str],
    ) -> str:
        """Pass 2: the sole LLM call. Authors or revises a scaffold's
        `content` -- never decides structure. One call per scaffold with
        changed evidence this sleep event, regardless of how many clusters
        it absorbed (callers group evidence by scaffold before calling)."""
        prompt = build_revise_strategy_prompt(
            current_summary=current_summary,
            positive_evidence=format_cluster_contents(positive_evidence),
            negative_evidence=format_cluster_contents(negative_evidence),
        )
        log_event(
            logger,
            "sleep_consolidation.revise_prompt",
            current_summary=current_summary,
            positive_count=len(positive_evidence),
            negative_count=len(negative_evidence),
            prompt=prompt,
        )
        response = self._generate(prompt, temperature=0)
        content = response.strip()
        if not content:
            raise ValueError("Empty strategy revision response")
        log_event(logger, "sleep_consolidation.revise_response", content=content)
        return content

    def consolidate(
        self,
        embeddings: Sequence[Sequence[float]],
        cluster_texts: Sequence[str],
        *,
        existing_scaffolds: Sequence[StrategicScaffoldContext] = (),
        stats_out: Optional[Dict[str, Any]] = None,
    ) -> List[SleepConsolidationResult]:
        """Cluster then decide structure (Pass 1) for each cluster.

        Content authoring (Pass 2) is not done here -- it happens per
        scaffold, after every cluster's structural decision has been applied
        (see `MemoryService.sleep_consolidate`), since one scaffold can
        absorb more than one cluster in the same sleep event and must get a
        single combined revision, not one overwriting the other.

        If `stats_out` is provided, it is populated in place with the RAW
        clustering stats (cluster_count, cluster_sizes, cluster_davies_bouldin)
        computed right after clustering, before any per-cluster decision can
        fail and get skipped -- and cluster_decision_failed_count, so "only
        1 cluster shows up" can be told apart from "clustering only formed 1
        cluster" vs. "clustering formed more, but decisions for the rest
        failed and were skipped." The returned list only contains clusters
        with a successful decision.
        """
        if len(embeddings) != len(cluster_texts):
            raise ValueError("embeddings and cluster_texts must have the same length")

        clusters = self.cluster_embeddings(embeddings)
        if stats_out is not None:
            stats_out["cluster_count"] = len(clusters)
            stats_out["cluster_sizes"] = [len(indices) for indices in clusters]
            stats_out["cluster_davies_bouldin"] = compute_davies_bouldin_index(embeddings, clusters)
            stats_out["cluster_decision_failed_count"] = 0

        results: List[SleepConsolidationResult] = []
        for indices in clusters:
            texts = [cluster_texts[idx] for idx in indices]
            cluster_embeds = [embeddings[idx] for idx in indices]
            try:
                decision = self.decide_cluster_structure(
                    cluster_embeds,
                    existing_scaffolds=existing_scaffolds,
                )
            except Exception as exc:
                # A structural-decision failure (e.g. an embedding-dimension
                # mismatch) must not take down the whole sleep-consolidation
                # event -- and by extension the whole training run. Skip
                # this cluster; its nodes stay unconsolidated and are
                # eligible again next time sleep consolidation fires.
                if stats_out is not None:
                    stats_out["cluster_decision_failed_count"] = (
                        stats_out.get("cluster_decision_failed_count", 0) + 1
                    )
                log_event(
                    logger,
                    "sleep_consolidation.cluster_decision_failed",
                    cluster_indices=list(indices),
                    error=str(exc),
                )
                logger.warning(
                    "Sleep consolidation cluster decision failed; skipping cluster "
                    "(nodes remain unconsolidated): %s",
                    exc,
                )
                continue
            results.append(
                SleepConsolidationResult(
                    cluster_indices=list(indices),
                    cluster_texts=texts,
                    action=decision.action,
                    target_scaffold_id=decision.target_scaffold_id,
                )
            )
        return results

    def _generate(self, prompt: str, **kwargs: object) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.llm_provider.generate(messages, **kwargs)
