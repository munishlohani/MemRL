"""Similarity retrieval for skill representations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
from math import sqrt
import math

from memrl.utils.q_utils import get_expected_option_value, get_q_salience


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


def rank_normalize(values: Sequence[float]) -> List[float]:
    """Map values to [0, 1] by rank within the sequence (ties share the
    average rank), so terms on incomparable raw scales -- e.g. an advantage
    centered at 0 with roughly half its mass negative, vs. a cosine
    similarity confined to [0, 1] -- can be blended without one silently
    dominating just because of its native magnitude (spec §P2.1)."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return [rank / (n - 1) for rank in ranks]


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

    @staticmethod
    def _tactical_task_value(node: Any, task_type_dominant: Optional[str], lambda_shrink: float) -> float:
        """Q_i(t_k) for the current task type (spec §3.6/§9.2); falls back to
        the cross-task shrinkage-weighted mean Q_bar_w when this node has
        never been used on t_k, so it stays selectable rather than locked out."""
        q_values = getattr(node, "Q", None) or {}
        if task_type_dominant is not None and task_type_dominant in q_values:
            return float(q_values[task_type_dominant])
        return get_q_salience(node, lambda_shrink=lambda_shrink)

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
        cluster_scoped: bool = False,
        task_type_dominant: Optional[str] = None,
        lambda_retrieval: float = 0.5,
        theta_retrieval: float = 0.0,
    ) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
        """Within-cluster retrieval blends per-task-type advantage with query
        similarity as a convex combination (spec §P2.1, tuned per §P2.8):

            score = lambda_retrieval * rank_norm(Q_i(t_k))
                    + (1 - lambda_retrieval) * rank_norm(cos(e_i, e_q))

        Both terms are rank-normalized over the candidate set first -- Q is
        an advantage centered near 0 (often negative), cosine similarity
        lives in [0, 1], so blending the raw values would let similarity
        dominate every comparison regardless of lambda_retrieval. The
        bootstrap fallback (cluster_scoped=False, no d=1 scaffold yet) keeps
        the decay-weighted pure-similarity score -- newly formed nodes
        rarely have usable per-task-type Q yet.

        theta_retrieval is a tactical-only quality gate. In cluster-scoped
        mode it is an ABSOLUTE cosine-similarity floor applied *before*
        rank-normalization: a candidate whose raw similarity to the current
        context is below it is dropped, so a step can legitimately retrieve
        nothing when no child of the active scaffold is relevant here -- no
        memory is better than a bad one. It must read raw similarity, not the
        blended score, because rank-normalization is set-relative and always
        maps the best candidate to ~1.0 -- a gate on the blended score can
        never drop the top pick and never fires when every candidate is
        globally irrelevant. In the bootstrap path the gate applies to the
        decay-weighted similarity score, which is already an absolute [0, 1]
        quantity.
        """
        rep_by_id = {getattr(rep, "id", None): rep for rep in representations}
        candidates: List[Dict[str, Any]] = []

        for node in nodes:
            if getattr(node, "depth", None) != 2:
                continue

            rep = rep_by_id.get(getattr(node, "id", None))
            if rep is None:
                continue

            last_accessed_step = int(getattr(node, "last_accessed_step", 0) or 0)
            decay_rate = float(getattr(node, "decay_rate", 0.0) or 0.0)
            delta_t = max(0, int(current_step) - last_accessed_step)
            decay_factor = math.exp(-decay_rate * float(delta_t))
            similarity = cosine_similarity(query_embedding, getattr(rep, "embedding", None))

            if not cluster_scoped and threshold is not None and similarity < float(threshold):
                continue

            q_value = (
                self._tactical_task_value(node, task_type_dominant, lambda_shrink)
                if cluster_scoped
                else get_q_salience(node, lambda_shrink=lambda_shrink)
            )
            candidates.append(
                {
                    "node": node,
                    "rep": rep,
                    "similarity": similarity,
                    "decay_factor": decay_factor,
                    "q_value": q_value,
                }
            )

        if cluster_scoped:
            # Absolute relevance floor: drop candidates whose raw cosine
            # similarity is below theta_retrieval BEFORE rank-normalizing, so
            # the gate can actually fire (return nothing) when no child of the
            # active scaffold is relevant to this step. Gating on the blended
            # rank-normalized score cannot -- the top candidate always ranks
            # ~1.0 regardless of absolute relevance.
            candidates = [
                c for c in candidates
                if float(c["similarity"]) >= float(theta_retrieval)
            ]
            lam = float(lambda_retrieval)
            q_norms = rank_normalize([c["q_value"] for c in candidates])
            sim_norms = rank_normalize([c["similarity"] for c in candidates])
            for candidate, q_norm, sim_norm in zip(candidates, q_norms, sim_norms):
                candidate["score"] = lam * q_norm + (1.0 - lam) * sim_norm
        else:
            for candidate in candidates:
                candidate["score"] = candidate["similarity"] * candidate["decay_factor"]
            # Bootstrap path: score = sim * decay_factor is already an
            # absolute [0, 1] quantity, so the same floor applies to it directly.
            candidates = [
                c for c in candidates if float(c["score"]) >= float(theta_retrieval)
            ]

        selected = [
            self._format_selected_payload(
                node=candidate["node"],
                representation=candidate["rep"],
                similarity=candidate["similarity"],
                score=candidate["score"],
                q_estimate=candidate["q_value"],
                decay_factor=candidate["decay_factor"],
            )
            for candidate in candidates
        ]

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
        query_embedding: Any = None,
        nodes: Sequence[Any],
        representations: Sequence[Any],
        top_k: int = 5,
        task_type_dominant: Optional[str] = None,
        lambda_retrieval: float = 0.5,
        ucb_c: float = 0.0,
    ) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
        """Strategic scaffold scoring uses the same convex blend as tactical
        retrieval (spec §P2.1, extended to §9.1 per explicit instruction),
        with the same lambda_retrieval -- one shared coefficient, not a
        second one:

            score = lambda_retrieval * rank_norm(q_i)
                    + (1 - lambda_retrieval) * rank_norm(cos(e_i, e_q))

        where q_i is Q_omega optionally boosted by a UCB1-style exploration
        term (guards against deterministic-argmax option starvation, spec
        §16, FeUdal/Option-Critic):

            q_i = Q_omega(t_k) + ucb_c * sqrt(ln(N + 1) / (n_i + 1))

        n_i is this scaffold's n_omega[t_k] (falling back to its total
        cross-task visit count when t_k is cold, mirroring Q_omega's own
        fallback); N is the sum of n_i over the candidate set. ucb_c=0.0
        (default) makes q_i == Q_omega exactly, recovering the prior
        pure-Q ranking. The +1 smoothing on both the log and the
        denominator avoids a div-by-zero / log(0) special case for a
        never-visited scaffold while still giving it the largest bonus
        in the set.

        No quality gate here (theta_retrieval is tactical-only) -- an
        episode must always end up with an active scaffold or explicit
        bootstrap-null, never a mid-episode "no scaffold" default.
        """
        rep_by_id = {getattr(rep, "id", None): rep for rep in representations}
        candidates: List[Dict[str, Any]] = []

        for node in nodes:
            if getattr(node, "depth", None) != 1:
                continue

            q_estimate = get_expected_option_value(node, task_type_dominant)
            rep = rep_by_id.get(getattr(node, "id", None))
            similarity = (
                cosine_similarity(query_embedding, getattr(rep, "embedding", None))
                if rep is not None
                else 0.0
            )
            n_omega = getattr(node, "n_omega", None) or {}
            if task_type_dominant is not None and task_type_dominant in n_omega:
                visit_count = float(n_omega[task_type_dominant])
            else:
                visit_count = float(sum(n_omega.values())) if n_omega else 0.0
            candidates.append(
                {
                    "node": node,
                    "rep": rep,
                    "q_value": q_estimate,
                    "similarity": similarity,
                    "visit_count": visit_count,
                }
            )

        total_visits = sum(c["visit_count"] for c in candidates)
        for candidate in candidates:
            ucb_bonus = float(ucb_c) * sqrt(
                math.log(total_visits + 1.0) / (candidate["visit_count"] + 1.0)
            )
            candidate["ucb_bonus"] = ucb_bonus
            candidate["ranking_q"] = candidate["q_value"] + ucb_bonus

        lam = float(lambda_retrieval)
        q_norms = rank_normalize([c["ranking_q"] for c in candidates])
        sim_norms = rank_normalize([c["similarity"] for c in candidates])
        for candidate, q_norm, sim_norm in zip(candidates, q_norms, sim_norms):
            candidate["score"] = lam * q_norm + (1.0 - lam) * sim_norm

        selected = [
            self._format_selected_payload(
                node=candidate["node"],
                representation=candidate["rep"],
                similarity=candidate["similarity"],
                score=candidate["score"],
                q_estimate=candidate["q_value"],
                decay_factor=1.0,
                ucb_bonus=candidate["ucb_bonus"],
            )
            for candidate in candidates
        ]

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
    def _format_selected_payload(
        *,
        node: Any,
        representation: Any,
        similarity: float,
        score: float,
        q_estimate: float,
        decay_factor: float,
        ucb_bonus: float = 0.0,
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
            "ucb_bonus": float(ucb_bonus),
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


__all__ = ["SkillSimilarityRetriever", "cosine_similarity", "rank_normalize"]
