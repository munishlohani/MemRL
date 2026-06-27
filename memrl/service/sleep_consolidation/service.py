"""Sleep consolidation orchestration."""

from __future__ import annotations

import json
from typing import List, Optional, Sequence, Tuple

from ...providers.base import BaseLLM
from ..strategies import ClusterStrategy
from .clustering import ClusteringStrategyBase, get_clustering_strategy
from .prompts import (
    build_sleep_consolidation_prompt,
    format_cluster_contents,
    format_existing_scaffolds,
)
from .types import (
    SleepConsolidationAction,
    SleepConsolidationDecision,
    SleepConsolidationResult,
    StrategicScaffoldContext,
)


class SleepConsolidationService:
    """LLM-backed sleep consolidation service.

    The LLM makes one structured decision per cluster: spawn, absorb, or discard.
    Structural node creation and Q_omega computation remain in code.
    """

    def __init__(
        self,
        llm_provider: BaseLLM,
        clustering_strategy: Optional[ClusteringStrategyBase] = None,
        *,
        cluster_strategy: ClusterStrategy = ClusterStrategy.KMEANS,
        n_clusters: Optional[int] = None,
        random_state: int = 0,
    ) -> None:
        self.llm_provider = llm_provider
        self.clustering_strategy = clustering_strategy or get_clustering_strategy(
            cluster_strategy,
            n_clusters=n_clusters,
            random_state=random_state,
        )

    def cluster_embeddings(
        self,
        embeddings: Sequence[Sequence[float]],
    ) -> List[List[int]]:
        """Group candidate embeddings into clusters."""
        return self.clustering_strategy.cluster(embeddings)

    def decide_cluster(
        self,
        cluster_texts: Sequence[str],
        *,
        existing_scaffolds: Sequence[StrategicScaffoldContext] = (),
    ) -> Tuple[SleepConsolidationDecision, str, str]:
        """Ask the LLM to choose spawn, absorb, or discard for one cluster."""
        cluster_contents = format_cluster_contents(cluster_texts)
        scaffold_contents = format_existing_scaffolds(existing_scaffolds)
        prompt = build_sleep_consolidation_prompt(
            cluster_contents=cluster_contents,
            existing_scaffolds=scaffold_contents,
        )
        raw_response = self._generate(prompt, temperature=0)
        decision = self._parse_decision(raw_response)
        if decision.action == SleepConsolidationAction.ABSORB:
            scaffold_ids = {scaffold.node_id for scaffold in existing_scaffolds}
            if decision.target_scaffold_id not in scaffold_ids:
                raise ValueError(
                    "Absorb decision referenced an unknown scaffold id: "
                    f"{decision.target_scaffold_id!r}"
                )
        return decision, prompt, raw_response

    def consolidate(
        self,
        embeddings: Sequence[Sequence[float]],
        cluster_texts: Sequence[str],
        *,
        existing_scaffolds: Sequence[StrategicScaffoldContext] = (),
    ) -> List[SleepConsolidationResult]:
        """Cluster then decide how to consolidate each cluster."""
        if len(embeddings) != len(cluster_texts):
            raise ValueError("embeddings and cluster_texts must have the same length")

        clusters = self.cluster_embeddings(embeddings)
        results: List[SleepConsolidationResult] = []
        for indices in clusters:
            texts = [cluster_texts[idx] for idx in indices]
            decision, prompt, raw_response = self.decide_cluster(
                texts,
                existing_scaffolds=existing_scaffolds,
            )
            results.append(
                SleepConsolidationResult(
                    cluster_indices=list(indices),
                    cluster_texts=texts,
                    action=decision.action,
                    summary=decision.summary,
                    target_scaffold_id=decision.target_scaffold_id,
                    prompt=prompt,
                    raw_response=raw_response,
                )
            )
        return results

    def _generate(self, prompt: str, **kwargs: object) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.llm_provider.generate(messages, **kwargs)

    @staticmethod
    def _parse_decision(response: str) -> SleepConsolidationDecision:
        payload = SleepConsolidationService._load_json_object(response)

        action_raw = payload.get("action")
        if not isinstance(action_raw, str):
            raise ValueError(f"Missing sleep-consolidation action: {response!r}")
        try:
            action = SleepConsolidationAction(action_raw.lower())
        except ValueError as exc:
            raise ValueError(f"Unrecognized sleep-consolidation action: {action_raw!r}") from exc

        summary = SleepConsolidationService._coerce_optional_str(payload.get("summary"))
        target_scaffold_id = SleepConsolidationService._coerce_optional_str(
            payload.get("target_scaffold_id")
        )

        if action == SleepConsolidationAction.SPAWN:
            if summary is None:
                raise ValueError(
                    "Spawn decisions must include a non-empty summary: "
                    f"{response!r}"
                )
            return SleepConsolidationDecision(
                action=action,
                summary=summary,
                target_scaffold_id=None,
            )

        if action == SleepConsolidationAction.ABSORB:
            if target_scaffold_id is None:
                raise ValueError(
                    "Absorb decisions must include target_scaffold_id: "
                    f"{response!r}"
                )
            return SleepConsolidationDecision(
                action=action,
                summary=None,
                target_scaffold_id=target_scaffold_id,
            )

        if action == SleepConsolidationAction.DISCARD:
            return SleepConsolidationDecision(
                action=action,
                summary=None,
                target_scaffold_id=None,
            )

        raise ValueError(f"Unsupported sleep-consolidation action: {action!r}")

    @staticmethod
    def _load_json_object(response: str) -> dict[str, object]:
        text = response.strip()
        if not text:
            raise ValueError("Empty sleep-consolidation response")

        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"Unable to parse sleep-consolidation response: {response!r}")
            loaded = json.loads(text[start : end + 1])

        if not isinstance(loaded, dict):
            raise ValueError(f"Sleep-consolidation response must be a JSON object: {response!r}")
        return loaded

    @staticmethod
    def _coerce_optional_str(value: object) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"Expected string or null in sleep-consolidation response: {value!r}")
        stripped = value.strip()
        return stripped or None
