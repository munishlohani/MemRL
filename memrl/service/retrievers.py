"""Similarity retrieval for skill representations."""

from __future__ import annotations

from math import sqrt
from typing import Any, Dict, List, Optional


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


__all__ = ["SkillSimilarityRetriever", "cosine_similarity"]
