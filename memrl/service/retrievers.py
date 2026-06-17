"""
Retrievers for different retrieve strategies in the Memp system.

This module provides a strategy-pattern implementation for retrieval logic,
mirroring builders.py for build strategies. It centralizes:
- Flattening MemOS search/get_all results
- Formatting each memory item into a consistent dict
- Concrete retrievers for RANDOM and QUERY
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import random

try:
    from memos.mem_os.main import MOS
    from memos.memories.textual.item import TextualMemoryItem
except Exception:  # pragma: no cover - optional legacy dependency
    MOS = Any  # type: ignore[assignment]
    TextualMemoryItem = Any  # type: ignore[assignment]

from .strategies import RetrieveStrategy


def embedding_similarity(left: Any, right: Any) -> float:
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
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))


class SkillSimilarityRetriever:
    """Retriever-style embedding search for skill nodes."""

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
            score = embedding_similarity(query_embedding, getattr(node, "embedding", None))
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



# ---------- Utilities ----------

def _extract_item_and_score(hit: Any) -> Tuple[Optional[TextualMemoryItem], Optional[float]]:
    """Normalize various hit shapes into (TextualMemoryItem, score).

    Supported shapes:
    - TextualMemoryItem (score=None)
    - {'item': TextualMemoryItem, 'score': float}
    - qdrant-like hit with attributes: payload (dict) and score (float)
    """
    if hit is None:
        return None, None

    # Dict hit: {'item': ..., 'score': ...}
    if isinstance(hit, dict) and "item" in hit:
        item = hit.get("item")
        score = hit.get("score")
        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None
        return item, score_f

    # Qdrant-like hit: has payload + score
    payload = getattr(hit, "payload", None)
    if isinstance(payload, dict):
        try:
            item = TextualMemoryItem(**payload)
        except Exception:
            item = None
        score = getattr(hit, "score", None)
        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None
        return item, score_f

    # Plain item
    if isinstance(hit, TextualMemoryItem) or hasattr(hit, "memory"):
        return hit, None

    return None, None


def _extract_full_content(metadata: Any) -> Optional[str]:
    """Best-effort extraction of full_content from metadata."""
    if metadata is None:
        return None
    try:
        # Pydantic v2: extra fields are in model_dump/model_extra
        if hasattr(metadata, "model_extra"):
            fc = metadata.model_extra.get("full_content")
            if isinstance(fc, str) and fc:
                return fc
    except Exception:
        pass
    try:
        if hasattr(metadata, "model_dump"):
            md = metadata.model_dump()
            fc = md.get("full_content")
            if isinstance(fc, str) and fc:
                return fc
    except Exception:
        pass
    try:
        fc = getattr(metadata, "full_content", None)
        if isinstance(fc, str) and fc:
            return fc
    except Exception:
        pass
    if isinstance(metadata, dict):
        fc = metadata.get("full_content")
        if isinstance(fc, str) and fc:
            return fc
    return None


def _extract_similarity_fallback(metadata: Any) -> float:
    """Fallback similarity when a hit has no explicit score."""
    # User decision: missing score is treated as 0.0 for filtering; keep same here.
    try:
        if metadata is None:
            return 0.0
        if isinstance(metadata, dict):
            return float(metadata.get("relativity", 0.0) or 0.0)
        # Some MemOS versions may expose relativity directly
        rel = getattr(metadata, "relativity", None)
        if rel is not None:
            return float(rel)
    except Exception:
        pass
    return 0.0


def _flatten_text_mem_results(result: Dict[str, Any]) -> List[TextualMemoryItem]:
    """Flatten MemOS MOSSearchResult text_mem section into a plain list of items.

    Expected input format (from MOS.search/get_all):
    {
        "text_mem": [
            {"cube_id": "...", "memories": [TextualMemoryItem, ...]},
            ...
        ],
        "act_mem": [...],
        "para_mem": [...],
    }
    """
    items: List[TextualMemoryItem] = []
    for cube in result.get("text_mem", []):
        items.extend(cube.get("memories", []))
    return items


def _format_memory_result(item: Any) -> Dict[str, Any]:
    """Format a TextualMemoryItem (or compatible dict) to a consistent dict.

    Returns keys:
    - memory_id: str
    - content: str (prefer metadata.full_content; fallback to item.memory)
    - metadata: Any
    - similarity: float (uses metadata.relativity if present)
    - memory_item: original object
    """
    mem_item, score = _extract_item_and_score(item)
    if mem_item is None:
        # Fail closed but keep schema stable.
        return {
            "memory_id": "unknown",
            "content": str(item),
            "metadata": {},
            "similarity": 0.0,
            "memory_item": item,
        }

    metadata = getattr(mem_item, "metadata", None)
    full_content = _extract_full_content(metadata)
    content = full_content or getattr(mem_item, "memory", "")
    similarity = float(score) if score is not None else _extract_similarity_fallback(metadata)

    return {
        "memory_id": getattr(mem_item, "id", "unknown"),
        "content": content,
        "metadata": metadata,
        "similarity": similarity,
        "memory_item": mem_item,
    }


# ---------- Strategy Base Class ----------

class BaseRetriever(ABC):
    def __init__(self, mos: MOS, user_id: str):
        self.mos = mos
        self.user_id = user_id

    @abstractmethod
    def retrieve(self, task_description: str, k: int, threshold: float) -> List[Dict[str, Any]]:
        ...


# ---------- Concrete Retrievers ----------

class RandomRetriever(BaseRetriever):
    def retrieve(self, task_description: str, k: int, threshold: float) -> List[Dict[str, Any]]:
        all_res = self.mos.get_all(user_id=self.user_id)
        items = _flatten_text_mem_results(all_res)
        if not items:
            return []
        sel = random.sample(items, min(k, len(items)))
        return [_format_memory_result(x) for x in sel]


class QueryRetriever(BaseRetriever):
    def retrieve(self, task_description: str, k: int, threshold: float) -> List[Dict[str, Any]]:
        # MemOS API compatibility: some versions don't provide search_score().
        search_score = getattr(self.mos, "search_score", None)
        if callable(search_score):
            res = search_score(query=task_description, user_id=self.user_id, top_k=k)
        else:
            search = getattr(self.mos, "search", None)
            if not callable(search):
                raise AttributeError("MOS has neither search_score() nor search()")
            res = search(query=task_description, user_id=self.user_id, top_k=k)
        items = _flatten_text_mem_results(res)
        out: List[Dict[str, Any]] = []
        for x in items:
            mem_item, score = _extract_item_and_score(x)
            # User decision: missing score is treated as 0.0 for filtering.
            sim = float(score) if score is not None else 0.0
            if threshold > 0 and sim < threshold:
                continue
            out.append(_format_memory_result(x))
            if len(out) >= k:
                break
        return out
# ---------- Factory ----------

def get_retriever(
    strategy: RetrieveStrategy,
    *,
    mos: MOS,
    user_id: str,
) -> BaseRetriever:
    if strategy == RetrieveStrategy.RANDOM:
        return RandomRetriever(mos, user_id)
    if strategy == RetrieveStrategy.QUERY:
        return QueryRetriever(mos, user_id)
    raise ValueError(f"Unsupported retrieve strategy: {strategy}")
    
