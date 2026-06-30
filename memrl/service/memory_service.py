"""Research-native memory service for the updated MemRL design."""

from __future__ import annotations

import json
import os
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from uuid import uuid4

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency fallback
    np = None
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    JSON,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..memory.graph import SkillGraph
from ..memory.skill_node import SkillNode
from ..memory.skill_representation import SkillRepresentation
from ..utils.q_utils import (
    compute_shrinkage_weighted_mean_from_samples,
    get_expected_option_value,
)
from .sleep_consolidation import (
    SleepConsolidationAction,
    SleepConsolidationResult,
    SleepConsolidationService,
    StrategicScaffoldContext,
)
from .retrievers import SkillSimilarityRetriever

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from ..configs.config import MemoryConfig
    from ..providers.base import BaseEmbedder
else:
    Engine = Any  # type: ignore[assignment]
    MemoryConfig = Any  # type: ignore[assignment]
    BaseEmbedder = Any  # type: ignore[assignment]


class MemoryService:
    """Small service wrapper around the skill graph and content index."""

    def __init__(
        self,
        memory_config: MemoryConfig,
        *,
        embedding_provider: Optional[BaseEmbedder] = None,
        graph: Optional[SkillGraph] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self.memory_config = memory_config
        self.embedding_provider = embedding_provider
        self.graph = graph or SkillGraph(
            lambda_base=memory_config.lambda_base,
            lambda_shrink=getattr(memory_config, "lambda_shrink", 10.0),
            epsilon=memory_config.epsilon_decay,
        )
        resolved_db_path = db_path or getattr(
            memory_config,
            "skill_db_path",
            "results/memrl/skill_memory.sqlite",
        )
        self.db_path = str(resolved_db_path)
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._engine = None
        self._metadata = None
        self.skill_representation_table = None
        self.skill_graph_state_table = None

        self._engine = create_engine(
            f"sqlite+pysqlite:///{Path(self.db_path).as_posix()}",
            future=True,
            connect_args={"check_same_thread": False},
        )
        self._metadata = MetaData()
        self.skill_representation_table = Table(
            "skill_representation",
            self._metadata,
            Column("node_id", String, primary_key=True),
            Column("content", Text, nullable=False),
            Column("embedding", LargeBinary, nullable=False),
        )
        self.skill_graph_state_table = Table(
            "skill_graph_state",
            self._metadata,
            Column("node_id", String, primary_key=True),
            Column("depth", Integer, nullable=False),
            Column("parent_id", String, nullable=True),
            Column("task_type_dominant", String, nullable=False),
            Column("t_create", Integer, nullable=False),
            Column("last_accessed_step", Integer, nullable=False),
            Column("decay_rate", Float, nullable=False),
            Column("consolidated", Boolean, nullable=False, default=False),
            Column("Q", JSON, nullable=False),
            Column("n", JSON, nullable=False),
            Column("Q_omega", JSON, nullable=False),
            Column("n_omega", JSON, nullable=False),
            Column("secondary_parents", JSON, nullable=False),
            Column("evidence_ids", JSON, nullable=False),
        )
        self._metadata.create_all(self._engine)

        if self.graph.nodes:
            self._sync_graph_state_to_db()
        else:
            self._load_graph_state()
        self.retriever = SkillSimilarityRetriever()

        # Empirical episode-length statistics per task type, used by the
        # finite-horizon Q^Omega initialization for spawned scaffolds
        # (spec §3.5, W3). Running mean to keep memory O(#task types).
        self._episode_length_sum: Dict[str, float] = {}
        self._episode_length_count: Dict[str, int] = {}

    def record_episode_length(self, task_type: str, length: int) -> None:
        """Feed an observed episode length into the per-task-type running mean.

        Used by the finite-horizon Q^Omega initialization. Call once per
        finished episode. No-op when ``length`` is not a positive int.
        """
        try:
            length_f = float(int(length))
        except (TypeError, ValueError):
            return
        if length_f <= 0.0 or not task_type:
            return
        self._episode_length_sum[task_type] = (
            self._episode_length_sum.get(task_type, 0.0) + length_f
        )
        self._episode_length_count[task_type] = (
            self._episode_length_count.get(task_type, 0) + 1
        )

    def mean_episode_length(self, task_type: Optional[str]) -> Optional[float]:
        """Return the tracked mean episode length for a task type, or None."""
        if not task_type:
            return None
        count = self._episode_length_count.get(task_type, 0)
        if count <= 0:
            return None
        return self._episode_length_sum.get(task_type, 0.0) / float(count)

    @staticmethod
    def _serialize_embedding(embedding: List[float]) -> bytes:
        values = [float(value) for value in embedding]
        if np is None:
            return json.dumps(values).encode("utf-8")
        array = np.asarray(values, dtype=np.float32)
        return array.tobytes()

    @staticmethod
    def _deserialize_embedding(blob: Any) -> List[float]:
        if blob is None:
            return []
        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        if isinstance(blob, (bytes, bytearray)):
            if np is not None and len(blob) % np.dtype(np.float32).itemsize == 0:
                return np.frombuffer(blob, dtype=np.float32).astype(float).tolist()
            try:
                text = bytes(blob).decode("utf-8")
                data = json.loads(text)
                return [float(value) for value in data]
            except Exception:
                return []
        if isinstance(blob, str):
            try:
                data = json.loads(blob)
                return [float(value) for value in data]
            except Exception:
                return []
        try:
            data = json.loads(str(blob))
            return [float(value) for value in data]
        except Exception:
            return []

    @staticmethod
    def _serialize_json(value: Any) -> Any:
        return value

    @staticmethod
    def _deserialize_json(blob: Any, default: Any) -> Any:
        if blob is None:
            return default
        if isinstance(blob, (list, dict)):
            return blob
        if isinstance(blob, bytes):
            text = blob.decode("utf-8")
        else:
            text = str(blob)
        try:
            return json.loads(text)
        except Exception:
            return default

    def _upsert_representation(self, representation: SkillRepresentation) -> None:
        stmt = sqlite_insert(self.skill_representation_table).values(
            node_id=representation.id,
            content=representation.content,
            embedding=self._serialize_embedding(representation.embedding),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["node_id"],
            set_={
                "content": stmt.excluded.content,
                "embedding": stmt.excluded.embedding,
            },
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)

    def _upsert_graph_state(self, node: SkillNode) -> None:
        stmt = sqlite_insert(self.skill_graph_state_table).values(
            node_id=node.id,
            depth=node.depth,
            parent_id=node.parent_id,
            task_type_dominant=node.task_type_dominant,
            t_create=node.t_create,
            last_accessed_step=node.last_accessed_step,
            decay_rate=node.decay_rate,
            consolidated=bool(node.consolidated),
            Q=self._serialize_json(node.Q),
            n=self._serialize_json(node.n),
            Q_omega=self._serialize_json(node.Q_omega),
            n_omega=self._serialize_json(node.n_omega),
            secondary_parents=self._serialize_json(node.secondary_parents),
            evidence_ids=self._serialize_json(node.evidence_ids),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["node_id"],
            set_={
                "depth": stmt.excluded.depth,
                "parent_id": stmt.excluded.parent_id,
                "task_type_dominant": stmt.excluded.task_type_dominant,
                "t_create": stmt.excluded.t_create,
                "last_accessed_step": stmt.excluded.last_accessed_step,
                "decay_rate": stmt.excluded.decay_rate,
                "consolidated": stmt.excluded.consolidated,
                "Q": stmt.excluded.Q,
                "n": stmt.excluded.n,
                "Q_omega": stmt.excluded.Q_omega,
                "n_omega": stmt.excluded.n_omega,
                "secondary_parents": stmt.excluded.secondary_parents,
                "evidence_ids": stmt.excluded.evidence_ids,
            },
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)

    def persist_node_state(self, node_or_id: SkillNode | str) -> None:
        """Persist the current in-memory node state back to SQLite."""
        node = node_or_id if isinstance(node_or_id, SkillNode) else self.graph.get(node_or_id)
        self._upsert_graph_state(node)

    def _fetch_representation(self, node_id: str) -> SkillRepresentation:
        stmt = select(
            self.skill_representation_table.c.node_id,
            self.skill_representation_table.c.content,
            self.skill_representation_table.c.embedding,
        ).where(self.skill_representation_table.c.node_id == node_id)
        with self._engine.begin() as conn:
            row = conn.execute(stmt).mappings().first()
        if row is None:
            raise KeyError(node_id)
        return SkillRepresentation(
            id=row["node_id"],
            content=row["content"],
            embedding=self._deserialize_embedding(row["embedding"]),
        )

    def _fetch_graph_state(self, node_id: str) -> SkillNode:
        stmt = select(self.skill_graph_state_table).where(
            self.skill_graph_state_table.c.node_id == node_id
        )
        with self._engine.begin() as conn:
            row = conn.execute(stmt).mappings().first()
        if row is None:
            raise KeyError(node_id)
        return SkillNode(
            id=row["node_id"],
            task_type_dominant=row["task_type_dominant"],
            t_create=int(row["t_create"]),
            depth=int(row["depth"]),
            parent_id=row["parent_id"],
            secondary_parents=list(
                self._deserialize_json(row["secondary_parents"], [])
            ),
            last_accessed_step=int(row["last_accessed_step"]),
            Q=dict(self._deserialize_json(row["Q"], {})),
            n=dict(self._deserialize_json(row["n"], {})),
            Q_omega=dict(self._deserialize_json(row["Q_omega"], {})),
            n_omega=dict(self._deserialize_json(row["n_omega"], {})),
            decay_rate=float(row["decay_rate"]),
            evidence_ids=list(self._deserialize_json(row["evidence_ids"], [])),
            consolidated=bool(row["consolidated"]),
        )

    def _load_graph_state(self) -> None:
        stmt = select(self.skill_graph_state_table.c.node_id).order_by(
            self.skill_graph_state_table.c.depth.asc(),
            self.skill_graph_state_table.c.t_create.asc(),
            self.skill_graph_state_table.c.node_id.asc(),
        )
        with self._engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        if not rows:
            return

        pending = [row["node_id"] for row in rows]
        inserted = set(self.graph.nodes)

        while pending:
            progressed = False
            remaining: List[str] = []
            for node_id in pending:
                node = self._fetch_graph_state(node_id)
                parent_id = node.parent_id
                if parent_id not in (None, self.graph.root_id) and parent_id not in inserted:
                    remaining.append(node_id)
                    continue
                if node_id not in inserted:
                    self.graph.insert(node, parent_id=parent_id)
                    inserted.add(node_id)
                progressed = True
            if not progressed:
                raise RuntimeError("Unable to reconstruct skill graph from SQLite state")
            pending = remaining

    def _sync_graph_state_to_db(self) -> None:
        for node in self.graph.nodes.values():
            self._upsert_graph_state(node)

    def _fetch_representations(
        self,
        *,
        node_ids: Optional[List[str]] = None,
        depth: Optional[int] = None,
        task_type_dominant: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SkillRepresentation]:
        if node_ids is None:
            stmt = select(self.skill_graph_state_table.c.node_id)
            if depth is not None:
                stmt = stmt.where(self.skill_graph_state_table.c.depth == depth)
            if task_type_dominant is not None:
                stmt = stmt.where(
                    self.skill_graph_state_table.c.task_type_dominant == task_type_dominant
                )
            stmt = stmt.order_by(
                self.skill_graph_state_table.c.depth.asc(),
                self.skill_graph_state_table.c.t_create.asc(),
                self.skill_graph_state_table.c.node_id.asc(),
            )
            if limit is not None:
                stmt = stmt.limit(int(limit))
            with self._engine.begin() as conn:
                rows = conn.execute(stmt).mappings().all()
            node_ids = [row["node_id"] for row in rows]
        if not node_ids:
            return []

        stmt = select(
            self.skill_representation_table.c.node_id,
            self.skill_representation_table.c.content,
            self.skill_representation_table.c.embedding,
        ).where(self.skill_representation_table.c.node_id.in_(node_ids))
        with self._engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        by_id = {
            row["node_id"]: SkillRepresentation(
                id=row["node_id"],
                content=row["content"],
                embedding=self._deserialize_embedding(row["embedding"]),
            )
            for row in rows
        }
        return [by_id[node_id] for node_id in node_ids if node_id in by_id]

    @property
    def epsilon(self) -> float:
        return float(self.memory_config.epsilon_decay)

    def add_node(
        self,
        node: SkillNode,
        representation: SkillRepresentation,
        parent_id: Optional[str] = None,
    ) -> SkillNode:
        """Insert a node into the graph and its SQLite representation store."""
        if node.id != representation.id:
            raise ValueError("SkillNode id must match representation id")
        self.graph.insert(node, parent_id=parent_id)
        self._upsert_representation(representation)
        self._upsert_graph_state(node)
        return node

    def add_node_from_text(
        self,
        *,
        id: str,
        content: str,
        task_type_dominant: str,
        t_create: int,
        depth: int,
        parent_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        evidence_ids: Optional[List[str]] = None,
        last_accessed_step: Optional[int] = None,
    ) -> SkillNode:
        """Create a node from text and add it to the service."""
        if embedding is None:
            if self.embedding_provider is None:
                raise ValueError(
                    "embedding_provider is required when embedding is omitted"
                )
            embedding = self.embedding_provider.embed_single(content)

        if depth == 1:
            node = SkillNode.create_strategic(
                id=id,
                task_type_dominant=task_type_dominant,
                t_create=t_create,
                parent_id=parent_id,
                evidence_ids=list(evidence_ids or []),
            )
        elif depth == 2:
            node = SkillNode.create_tactical(
                id=id,
                task_type_dominant=task_type_dominant,
                t_create=t_create,
                parent_id=parent_id,
                evidence_ids=list(evidence_ids or []),
            )
        else:
            raise ValueError("depth must be 1 or 2")

        if last_accessed_step is not None:
            node.last_accessed_step = int(last_accessed_step)

        representation = SkillRepresentation(
            id=id,
            content=content,
            embedding=embedding,
        )
        return self.add_node(node, representation, parent_id=parent_id)

#O(n^2) time complexity. Need to work on this

    def prune_tactical_nodes(
        self,
        *,
        current_step: Optional[int] = None,
        theta_prune: Optional[float] = None,
    ) -> List[str]:
        """Prune tactical nodes whose decay-based retention falls below threshold."""
        resolved_threshold = theta_prune
        if resolved_threshold is None:
            resolved_threshold = getattr(self.memory_config, "theta_prune", None)
        if resolved_threshold is None:
            return []

        threshold = float(resolved_threshold)
        resolved_step = int(
            current_step
            if current_step is not None
            else getattr(self.graph, "current_step", 0) or 0
        )

        removed_ids: List[str] = []
        tactical_nodes = [
            node for node in list(self.graph.nodes.values()) if node.is_tactical
        ]
        for node in tactical_nodes:
            delta_t = max(0, resolved_step - int(node.last_accessed_step))
            retention = math.exp(-float(node.decay_rate) * float(delta_t))
            if retention < threshold:
                node_removed_ids = self.graph.remove(node.id)
                removed_ids.extend(node_removed_ids)
                with self._engine.begin() as conn:
                    for rid in node_removed_ids:
                        conn.execute(
                            delete(self.skill_representation_table).where(
                                self.skill_representation_table.c.node_id == rid
                            )
                        )
                        conn.execute(
                            delete(self.skill_graph_state_table).where(
                                self.skill_graph_state_table.c.node_id == rid
                            )
                        )

        return removed_ids

    def get_node(self, node_id: str) -> SkillNode:
        return self.graph.get(node_id)

    def get_representation(self, node_id: str) -> SkillRepresentation:
        return self._fetch_representation(node_id)

    def list_nodes(self, depth: Optional[int] = None) -> List[SkillNode]:
        nodes = list(self.graph.nodes.values())
        if depth is None:
            return nodes
        return [node for node in nodes if node.depth == depth]

    def list_representations(
        self,
        depth: Optional[int] = None,
        *,
        task_type_dominant: Optional[str] = None,
    ) -> List[SkillRepresentation]:
        return self._fetch_representations(
            depth=depth,
            task_type_dominant=task_type_dominant,
        )

    def search_nodes(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 5,
        depth: Optional[int] = None,
        task_type_dominant: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Similarity search over stored skill nodes."""
        return self.retriever.search(
            self._fetch_representations(
                depth=depth,
                task_type_dominant=task_type_dominant,
            ),
            query_embedding,
            top_k=top_k,
            depth=None,
        )

    def search(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 5,
        depth: Optional[int] = None,
        task_type_dominant: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Alias for embedding-based similarity search."""
        return self.search_nodes(
            query_embedding,
            top_k=top_k,
            depth=depth,
            task_type_dominant=task_type_dominant,
        )

    def search_by_text(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        depth: Optional[int] = None,
        task_type_dominant: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convenience wrapper that embeds text before similarity search."""
        if self.embedding_provider is None:
            raise ValueError("embedding_provider is required for text search")
        query_embedding = self.embedding_provider.embed_single(query_text)
        return self.search_nodes(
            query_embedding,
            top_k=top_k,
            depth=depth,
            task_type_dominant=task_type_dominant,
        )

    def find_best_parent(
        self,
        query_embedding: List[float],
        *,
        target_depth: int,
        task_type_dominant: Optional[str] = None,
    ) -> Optional[SkillNode]:
        """Find the best parent candidate for a query embedding."""
        best = self.retriever.best_node(
            self._fetch_representations(
                depth=target_depth,
                task_type_dominant=task_type_dominant,
            ),
            query_embedding,
            depth=None,
        )
        if best is None:
            return None
        return self.graph.get(best.id)

    def sleep_consolidation_count(self) -> int:
        """Return the active count used to decide whether sleep consolidation fires."""
        return self.graph.unabsorbed_tactical_count()

    def should_sleep_consolidate(self) -> bool:
        """Check whether the current graph should trigger sleep consolidation."""
        threshold = getattr(self.memory_config, "n_sleep", None)
        if threshold is None:
            return False
        return self.sleep_consolidation_count() >= int(threshold)

    def sleep_consolidate(
        self,
        consolidation_service: SleepConsolidationService,
        *,
        theta_consolidate: Optional[float] = None,
    ) -> List[SleepConsolidationResult]:
        """Run sleep consolidation and wire consolidation outcomes into the graph.

        This performs the graph mutation phase only: cluster eligible tactical nodes,
        ask the LLM for a structured spawn/absorb/discard action, and materialize a
        strategic scaffold node for clusters judged to spawn one. Tactical nodes
        processed in any branch are marked consolidated.
        """
        resolved_threshold = theta_consolidate
        if resolved_threshold is None:
            resolved_threshold = getattr(self.memory_config, "theta_consolidate", None)
        threshold = float(resolved_threshold) if resolved_threshold is not None else 0.0

        eligible_nodes = [
            node
            for node in self.graph.nodes.values()
            if node.is_tactical
            and not node.consolidated
            and node.q_salience(self.graph.lambda_shrink) > threshold
        ]
        eligible_nodes.sort(key=lambda node: (int(node.t_create), node.id))
        if not eligible_nodes:
            return []

        eligible_representations = {
            node.id: self._fetch_representation(node.id) for node in eligible_nodes
        }

        eligible_embeddings = [
            eligible_representations[node.id].embedding for node in eligible_nodes
        ]
        clusters = consolidation_service.cluster_embeddings(eligible_embeddings)
        existing_scaffolds = self._strategic_scaffold_contexts()

        results: List[SleepConsolidationResult] = []
        for indices in clusters:
            cluster_nodes = [eligible_nodes[idx] for idx in indices]
            if not cluster_nodes:
                continue

            cluster_texts = [
                eligible_representations[node.id].content for node in cluster_nodes
            ]
            decision, prompt, raw_response = consolidation_service.decide_cluster(
                cluster_texts,
                existing_scaffolds=existing_scaffolds,
            )
            result = SleepConsolidationResult(
                cluster_indices=list(indices),
                cluster_texts=cluster_texts,
                action=decision.action,
                summary=decision.summary,
                target_scaffold_id=decision.target_scaffold_id,
                prompt=prompt,
                raw_response=raw_response,
            )
            results.append(result)

            if decision.action == SleepConsolidationAction.SPAWN:
                if decision.summary is None:
                    raise ValueError("Spawn decisions must include a scaffold summary")
                scaffold_node = self._spawn_strategic_scaffold(
                    cluster_nodes=cluster_nodes,
                    cluster_embeddings=[
                        eligible_representations[node.id].embedding
                        for node in cluster_nodes
                    ],
                    scaffold_content=decision.summary,
                )
                for node in cluster_nodes:
                    self.graph.reparent(node, scaffold_node.id)
                    node.consolidated = True
                    self._upsert_graph_state(node)
            elif decision.action == SleepConsolidationAction.ABSORB:
                if decision.target_scaffold_id is None:
                    raise ValueError("Absorb decisions must include a target scaffold id")
                target_scaffold = self._resolve_strategic_scaffold(
                    decision.target_scaffold_id
                )
                for node in cluster_nodes:
                    self.graph.reparent(node, target_scaffold.id)
                    node.consolidated = True
                    self._upsert_graph_state(node)
            elif decision.action == SleepConsolidationAction.DISCARD:
                for node in cluster_nodes:
                    node.consolidated = True
                    self._upsert_graph_state(node)
            else:
                raise ValueError(
                    f"Unsupported sleep-consolidation action: {decision.action!r}"
                )

        return results

    def _strategic_scaffold_contexts(self) -> List[StrategicScaffoldContext]:
        scaffolds = sorted(
            self.graph.nodes_at_depth(1),
            key=lambda node: (int(node.t_create), node.id),
        )
        return [
            StrategicScaffoldContext(
                node_id=node.id,
                summary=self._fetch_representation(node.id).content,
            )
            for node in scaffolds
        ]

    def _resolve_strategic_scaffold(self, scaffold_id: str) -> SkillNode:
        node = self.graph.get(scaffold_id)
        if not node.is_strategic:
            raise ValueError(f"Target node is not a strategic scaffold: {scaffold_id}")
        return node

    def _spawn_strategic_scaffold(
        self,
        *,
        cluster_nodes: List[SkillNode],
        cluster_embeddings: List[List[float]],
        scaffold_content: str,
    ) -> SkillNode:
        scaffold_id = uuid4().hex
        task_type_dominant = self._majority_task_type(cluster_nodes)
        scaffold_embedding = self._scaffold_embedding(
            scaffold_content=scaffold_content,
            cluster_embeddings=cluster_embeddings,
        )
        scaffold_q_omega = self._spawned_scaffold_q_omega(cluster_nodes)
        scaffold_evidence_ids = self._merged_evidence_ids(cluster_nodes)

        scaffold_node = SkillNode.create_strategic(
            id=scaffold_id,
            task_type_dominant=task_type_dominant,
            t_create=int(self.graph.current_step),
            parent_id=self.graph.root_id,
            evidence_ids=scaffold_evidence_ids,
        )
        scaffold_node.Q_omega = scaffold_q_omega

        representation = SkillRepresentation(
            id=scaffold_id,
            content=scaffold_content,
            embedding=scaffold_embedding,
        )
        self.add_node(scaffold_node, representation, parent_id=self.graph.root_id)
        return scaffold_node

    def _scaffold_embedding(
        self,
        *,
        scaffold_content: str,
        cluster_embeddings: List[List[float]],
    ) -> List[float]:
        if self.embedding_provider is not None:
            return self.embedding_provider.embed_single(scaffold_content)
        return self._mean_embedding(cluster_embeddings)

    @staticmethod
    def _mean_embedding(cluster_embeddings: List[List[float]]) -> List[float]:
        if not cluster_embeddings:
            return []
        dim = min(len(row) for row in cluster_embeddings)
        if dim <= 0:
            return []
        totals = [0.0] * dim
        for row in cluster_embeddings:
            for idx in range(dim):
                totals[idx] += float(row[idx])
        count = float(len(cluster_embeddings))
        return [value / count for value in totals]

    def _spawned_scaffold_q_omega(self, cluster_nodes: List[SkillNode]) -> Dict[str, float]:
        gamma_omega = float(getattr(self.memory_config, "gamma_omega", 0.95))
        lambda_shrink = float(
            getattr(self.memory_config, "lambda_shrink", self.graph.lambda_shrink)
        )
        horizon_mode = str(
            getattr(self.memory_config, "q_omega_init_horizon", "infinite")
        ).lower()
        min_horizon = max(1, int(getattr(self.memory_config, "q_omega_init_min_horizon", 1)))
        infinite_scale = 1.0 / max(1e-12, 1.0 - gamma_omega)

        def _scale_for(task_type: str) -> float:
            # Finite-horizon geometric sum S(T) = (1 - gamma^T) / (1 - gamma).
            # Always <= 1/(1-gamma), so empirical init never exceeds the
            # infinite-horizon upper bound (spec §3.5 approximation note).
            if horizon_mode != "empirical":
                return infinite_scale
            mean_len = self.mean_episode_length(task_type)
            if mean_len is None:
                return infinite_scale
            horizon = max(min_horizon, int(round(mean_len)))
            return (1.0 - gamma_omega ** float(horizon)) / max(1e-12, 1.0 - gamma_omega)

        q_omega: Dict[str, float] = {}
        task_types = sorted({task_type for node in cluster_nodes for task_type in node.Q})
        for task_type in task_types:
            samples = [
                (float(node.Q[task_type]), int(node.n.get(task_type, 0) or 0))
                for node in cluster_nodes
                if task_type in node.Q and int(node.n.get(task_type, 0) or 0) > 0
            ]
            if not samples:
                continue
            q_omega[task_type] = _scale_for(task_type) * compute_shrinkage_weighted_mean_from_samples(
                samples,
                lambda_shrink=lambda_shrink,
            )
        return q_omega

    @staticmethod
    def _majority_task_type(cluster_nodes: List[SkillNode]) -> str:
        counts = Counter(node.task_type_dominant for node in cluster_nodes if node.task_type_dominant)
        if not counts:
            return "unknown"
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    @staticmethod
    def _merged_evidence_ids(cluster_nodes: List[SkillNode]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for node in cluster_nodes:
            for evidence_id in node.evidence_ids:
                if evidence_id in seen:
                    continue
                seen.add(evidence_id)
                merged.append(evidence_id)
        return merged

    def close(self) -> None:
        """Close the SQLite connection."""
        self._engine.dispose()

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its subtree from the graph and SQLite table."""
        removed_ids = self.graph.remove(node_id)
        with self._engine.begin() as conn:
            for rid in removed_ids:
                conn.execute(
                    delete(self.skill_representation_table).where(
                        self.skill_representation_table.c.node_id == rid
                    )
                )
                conn.execute(
                    delete(self.skill_graph_state_table).where(
                        self.skill_graph_state_table.c.node_id == rid
                    )
                )

    def refresh_content_db(self) -> None:
        """Synchronize both SQLite tables with the graph."""
        valid_ids = list(self.graph.nodes)
        with self._engine.begin() as conn:
            if not valid_ids:
                conn.execute(delete(self.skill_representation_table))
                conn.execute(delete(self.skill_graph_state_table))
                return

            conn.execute(
                delete(self.skill_representation_table).where(
                    ~self.skill_representation_table.c.node_id.in_(valid_ids)
                )
            )
            conn.execute(
                delete(self.skill_graph_state_table).where(
                    ~self.skill_graph_state_table.c.node_id.in_(valid_ids)
                )
            )

    def retrieve_query(
        self,
        query_text: str,
        *,
        k: int = 5,
        depth: Optional[int] = None,
        threshold: float = 0.0,
        current_step: Optional[int] = None,
        task_type_dominant: Optional[str] = None,
        active_strategic_node_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
        """Retrieve memories using the current graph state and legacy tuple output.

        Strategic retrieval selects the active d=1 scaffold first. Tactical retrieval
        is then scoped to that scaffold's direct children. If no strategic scaffold
        exists yet, fall back to a flat tactical scan for bootstrap behavior only.
        """
        if self.embedding_provider is None:
            raise ValueError("embedding_provider is required for text retrieval")

        query_embedding = self.embedding_provider.embed_single(query_text)
        nodes = list(self.graph.nodes.values())
        representations = self._fetch_representations(
            depth=1 if depth == 1 else 2,
        )

        if depth == 1:
            return self.retriever.strategic_retrieve(
                query_text=query_text,
                nodes=nodes,
                representations=representations,
                top_k=k,
                task_type_dominant=task_type_dominant,
            )

        if depth not in (None, 2):
            raise ValueError("retrieve_query only supports depth=1 or depth=2")

        active_scaffold = self._select_active_strategic_scaffold(
            task_type_dominant=task_type_dominant,
            forced_scaffold_id=active_strategic_node_id,
        )

        if active_scaffold is None:
            tactical_nodes = [
                node
                for node in nodes
                if getattr(node, "depth", None) == 2
            ]
            resolved_step = int(
                current_step
                if current_step is not None
                else getattr(self.graph, "current_step", 0) or 0
            )
            return self.retriever.tactical_retrieve(
                query_text=query_text,
                query_embedding=query_embedding,
                nodes=tactical_nodes,
                representations=representations,
                top_k=k,
                threshold=threshold,
                current_step=resolved_step,
                lambda_shrink=float(getattr(self.graph, "lambda_shrink", 10.0) or 10.0),
                cluster_scoped=False,
            )

        tactical_node_ids = self.graph.child_ids(active_scaffold.id)
        tactical_nodes = [
            node
            for node in nodes
            if getattr(node, "id", None) in tactical_node_ids
        ]

        resolved_step = int(current_step if current_step is not None else getattr(self.graph, "current_step", 0) or 0)
        tactical_result, topk_queries = self.retriever.tactical_retrieve(
            query_text=query_text,
            query_embedding=query_embedding,
            nodes=tactical_nodes,
            representations=representations,
            top_k=k,
            threshold=threshold,
            current_step=resolved_step,
            lambda_shrink=float(getattr(self.graph, "lambda_shrink", 10.0) or 10.0),
            cluster_scoped=True,
        )
        tactical_result["active_strategic_node_id"] = active_scaffold.id
        tactical_result["active_strategic_score"] = float(
            get_expected_option_value(active_scaffold, task_type_dominant)
        )
        return tactical_result, topk_queries

    def _select_active_strategic_scaffold(
        self,
        *,
        task_type_dominant: Optional[str],
        forced_scaffold_id: Optional[str] = None,
    ) -> Optional[SkillNode]:
        if forced_scaffold_id is not None:
            return self.graph.get(forced_scaffold_id)

        strategic_nodes = self.graph.nodes_at_depth(1)
        if not strategic_nodes:
            return None

        def _score(node: SkillNode) -> float:
            return float(get_expected_option_value(node, task_type_dominant))

        return max(
            strategic_nodes,
            key=lambda node: (
                _score(node),
                int(node.t_create),
                node.id,
            ),
        )
