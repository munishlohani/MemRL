"""Similarity retrieval for skill representations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
from math import sqrt
import math


def cosine_similarity(left: Any, right: Any) -> float:
    """Compute cosine similarity between two embedding vectors."""

    def _as_list(value: Any) -> List[float]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return [float(x) for x in value]
        try:
            return [float(x) for x in list(value)]
        except Exception:
            return []

    left_vec = _as_list(left)
    right_vec = _as_list(right)
    if not left_vec or not right_vec:
        return 0.0

    size = min(len(left_vec), len(right_vec))
    if size == 0:
        return 0.0

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for idx in range(size):
        l_val = left_vec[idx]
        r_val = right_vec[idx]
        dot += l_val * r_val
        left_norm += l_val * l_val
        right_norm += r_val * r_val

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (sqrt(left_norm) * sqrt(right_norm))


class SkillSimilarityRetriever:
    """Retriever-style embedding search for skill representations."""

    def rank_nodes(
        self,
        nodes: List[Any],
        query_embedding: Any,
        top_k: int = 5,
        depth: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for node in nodes:
            node_depth = getattr(node, "depth", None)
            if depth is not None and node_depth != depth:
                continue
            score = cosine_similarity(query_embedding, getattr(node, "embedding", None))
            ranked.append(
                {
                    "node": node,
                    "node_id": getattr(node, "id", None),
                    "content": getattr(node, "content", None),
                    "depth": node_depth,
                    "score": score,
                }
            )
        ranked.sort(
            key=lambda item: (item["score"], item.get("node_id") or ""),
            reverse=True,
        )
        return ranked[:top_k]

    def search(
        self,
        nodes: List[Any],
        query_embedding: Any,
        top_k: int = 5,
        depth: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Alias for rank_nodes to keep similarity search self-contained."""
        return self.rank_nodes(
            nodes,
            query_embedding,
            top_k=top_k,
            depth=depth,
        )

    def best_node(
        self,
        nodes: List[Any],
        query_embedding: Any,
        depth: Optional[int] = None,
    ) -> Optional[Any]:
        ranked = self.rank_nodes(nodes, query_embedding, top_k=1, depth=depth)
        if not ranked:
            return None
        return ranked[0]["node"]

    def tactical_retrieve(
        self,
        *,
        query_text: str,
        query_embedding: Any,
        nodes: Sequence[Any],
        representations: Sequence[Any],
        top_k: int = 5,
        threshold: float = 0.0,
        current_step: int = 0,
        lambda_shrink: float = 10.0,
    ) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
        rep_by_id = {getattr(rep, "id", None): rep for rep in representations}
        selected: List[Dict[str, Any]] = []

        for node in nodes:
            if getattr(node, "depth", None) != 2:
                continue

            rep = rep_by_id.get(getattr(node, "id", None))
            if rep is None:
                continue

            similarity = cosine_similarity(query_embedding, getattr(rep, "embedding", None))
            if threshold is not None and similarity < float(threshold):
                continue

            last_accessed_step = int(getattr(node, "last_accessed_step", 0) or 0)
            decay_rate = float(getattr(node, "decay_rate", 0.0) or 0.0)
            delta_t = max(0, int(current_step) - last_accessed_step)
            decay_factor = math.exp(-decay_rate * float(delta_t))
            q_value = self._weighted_mean_q(node, lambda_shrink=lambda_shrink)
            score = similarity * decay_factor

            selected.append(
                self._format_selected_payload(
                    node=node,
                    representation=rep,
                    similarity=similarity,
                    score=score,
                    q_estimate=q_value,
                    decay_factor=decay_factor,
                )
            )

        selected.sort(
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                float(item.get("similarity", 0.0) or 0.0),
                str(item.get("memory_id") or ""),
            ),
            reverse=True,
        )
        selected = selected[: max(0, int(top_k))]
        simmax = max((float(item.get("similarity", 0.0) or 0.0) for item in selected), default=0.0)
        topk_queries = [(query_text, 1.0)] if query_text else []
        return {"selected": selected, "simmax": simmax}, topk_queries

    def strategic_retrieve(
        self,
        *,
        query_text: str,
        nodes: Sequence[Any],
        representations: Sequence[Any],
        top_k: int = 5,
        task_type_dominant: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
        rep_by_id = {getattr(rep, "id", None): rep for rep in representations}
        selected: List[Dict[str, Any]] = []

        for node in nodes:
            if getattr(node, "depth", None) != 1:
                continue

            q_estimate = self._expected_option_value(node, task_type_dominant)
            rep = rep_by_id.get(getattr(node, "id", None))
            selected.append(
                self._format_selected_payload(
                    node=node,
                    representation=rep,
                    similarity=0.0,
                    score=q_estimate,
                    q_estimate=q_estimate,
                    decay_factor=1.0,
                )
            )

        selected.sort(
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                str(item.get("memory_id") or ""),
            ),
            reverse=True,
        )
        selected = selected[: max(0, int(top_k))]
        simmax = max((float(item.get("score", 0.0) or 0.0) for item in selected), default=0.0)
        topk_queries = [(task_type_dominant or query_text, 1.0)] if (task_type_dominant or query_text) else []
        return {"selected": selected, "simmax": simmax}, topk_queries

    @staticmethod
    def _weighted_mean_q(node: Any, lambda_shrink: float = 10.0) -> float:
        q_values = getattr(node, "Q", None) or {}
        if not q_values:
            return 0.0

        n_values = getattr(node, "n", None) or {}
        weighted_sum = 0.0
        weight_sum = 0.0
        for task_type, q_value in q_values.items():
            count = float(n_values.get(task_type, 0) or 0)
            weight = count / (count + float(lambda_shrink))
            weighted_sum += weight * float(q_value)
            weight_sum += weight
        if weight_sum == 0.0:
            return 0.0
        return weighted_sum / weight_sum

    @staticmethod
    def _expected_option_value(node: Any, task_type_dominant: Optional[str] = None) -> float:
        q_omega = getattr(node, "Q_omega", None) or {}
        if task_type_dominant is not None and task_type_dominant in q_omega:
            return float(q_omega.get(task_type_dominant, 0.0) or 0.0)
        if not q_omega:
            return 0.0

        n_omega = getattr(node, "n_omega", None) or {}
        total_counts = float(sum(int(v) for v in n_omega.values()))
        if total_counts <= 0.0:
            return 0.0

        expected = 0.0
        for task_type, q_value in q_omega.items():
            weight = float(n_omega.get(task_type, 0) or 0) / total_counts
            expected += weight * float(q_value)
        return expected

    @staticmethod
    def _format_selected_payload(
        *,
        node: Any,
        representation: Any,
        similarity: float,
        score: float,
        q_estimate: float,
        decay_factor: float,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "memory_id": getattr(node, "id", None),
            "id": getattr(node, "id", None),
            "content": getattr(representation, "content", ""),
            "depth": int(getattr(node, "depth", 0) or 0),
            "parent_id": getattr(node, "parent_id", None),
            "task_type_dominant": getattr(node, "task_type_dominant", None),
            "t_create": int(getattr(node, "t_create", 0) or 0),
            "last_accessed_step": int(getattr(node, "last_accessed_step", 0) or 0),
            "decay_rate": float(getattr(node, "decay_rate", 0.0) or 0.0),
            "decay_factor": float(decay_factor),
            "similarity": float(similarity),
            "score": float(score),
            "q_estimate": float(q_estimate),
            "consolidated": bool(getattr(node, "consolidated", False)),
            "Q": dict(getattr(node, "Q", {}) or {}),
            "n": dict(getattr(node, "n", {}) or {}),
            "Q_omega": dict(getattr(node, "Q_omega", {}) or {}),
            "n_omega": dict(getattr(node, "n_omega", {}) or {}),
            "secondary_parents": list(getattr(node, "secondary_parents", []) or []),
            "evidence_ids": list(getattr(node, "evidence_ids", []) or []),
        }
        if representation is not None:
            payload["embedding"] = list(getattr(representation, "embedding", []) or [])
        return payload


__all__ = ["SkillSimilarityRetriever", "cosine_similarity"]
