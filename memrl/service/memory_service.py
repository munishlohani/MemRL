"""Research-native memory service for the updated MemRL design."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..memory.graph import SkillGraph
from ..memory.skill_node import SkillNode
from ..memory.skill_representation import SkillRepresentation
from .retrievers import SkillSimilarityRetriever

if TYPE_CHECKING:
    from ..configs.config import MemoryConfig
    from ..providers.base import BaseEmbedder
else:
    MemoryConfig = Any  # type: ignore[assignment]
    BaseEmbedder = Any  # type: ignore[assignment]


class MemoryService:
    """Small service wrapper around the skill graph and content index.

    The legacy MemOS-oriented parameters and storage plumbing were removed on
    purpose. This service keeps the research-facing objects only:
    - `MemoryConfig` for hyperparameters
    - `SkillGraph` for structure
    - SQLite `skill_representation` for the write-once content/embedding payloads
    - `SkillSimilarityRetriever` for similarity search
    """

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
            lambda_slow=memory_config.lambda_slow,
            lambda_fast=memory_config.effective_lambda_fast,
            epsilon=memory_config.epsilon_decay,
        )
        resolved_db_path = db_path or getattr(
            memory_config,
            "skill_db_path",
            "results/memrl/skill_memory.sqlite",
        )
        self.db_path = resolved_db_path
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._db = sqlite3.connect(self.db_path)
        self._db.row_factory = sqlite3.Row
        self._init_schema()
        if self.graph.nodes:
            self._sync_graph_state_to_db()
        else:
            self._load_graph_state()
        self.retriever = SkillSimilarityRetriever()

    def _init_schema(self) -> None:
        with self._db:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_representation (
                    node_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_graph_state (
                    node_id TEXT PRIMARY KEY,
                    depth INTEGER NOT NULL,
                    parent_id TEXT,
                    task_type_primary TEXT NOT NULL,
                    t_create INTEGER NOT NULL,
                    last_accessed_step INTEGER NOT NULL,
                    decay_rate REAL NOT NULL,
                    absorbed_by_sleep INTEGER NOT NULL,
                    Q TEXT NOT NULL,
                    n TEXT NOT NULL,
                    Q_omega TEXT NOT NULL,
                    n_omega TEXT NOT NULL,
                    secondary_parents TEXT NOT NULL,
                    evidence_ids TEXT NOT NULL
                )
                """
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_skill_graph_parent ON skill_graph_state(parent_id)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_skill_graph_depth ON skill_graph_state(depth)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_skill_graph_task_depth ON skill_graph_state(task_type_primary, depth)"
            )

    @staticmethod
    def _serialize_embedding(embedding: List[float]) -> bytes:
        payload = [float(value) for value in embedding]
        return json.dumps(payload).encode("utf-8")

    @staticmethod
    def _serialize_json(value: Any) -> str:
        return json.dumps(value, separators=(",", ":"))

    @staticmethod
    def _deserialize_embedding(blob: Any) -> List[float]:
        if blob is None:
            return []
        if isinstance(blob, bytes):
            text = blob.decode("utf-8")
        elif isinstance(blob, str):
            text = blob
        else:
            text = str(blob)
        try:
            data = json.loads(text)
            return [float(value) for value in data]
        except Exception:
            return []

    def _upsert_representation(self, representation: SkillRepresentation) -> None:
        with self._db:
            self._db.execute(
                """
                INSERT INTO skill_representation (node_id, content, embedding)
                VALUES (?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    content = excluded.content,
                    embedding = excluded.embedding
                """,
                (
                    representation.id,
                    representation.content,
                    self._serialize_embedding(representation.embedding),
                ),
            )

    def _upsert_graph_state(self, node: SkillNode) -> None:
        with self._db:
            self._db.execute(
                """
                INSERT INTO skill_graph_state (
                    node_id, depth, parent_id, task_type_primary, t_create,
                    last_accessed_step, decay_rate, absorbed_by_sleep, Q, n,
                    Q_omega, n_omega, secondary_parents, evidence_ids
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    depth = excluded.depth,
                    parent_id = excluded.parent_id,
                    task_type_primary = excluded.task_type_primary,
                    t_create = excluded.t_create,
                    last_accessed_step = excluded.last_accessed_step,
                    decay_rate = excluded.decay_rate,
                    absorbed_by_sleep = excluded.absorbed_by_sleep,
                    Q = excluded.Q,
                    n = excluded.n,
                    Q_omega = excluded.Q_omega,
                    n_omega = excluded.n_omega,
                    secondary_parents = excluded.secondary_parents,
                    evidence_ids = excluded.evidence_ids
                """,
                (
                    node.id,
                    node.depth,
                    node.parent_id,
                    node.task_type_primary,
                    node.t_create,
                    node.last_accessed_step,
                    node.decay_rate,
                    1 if node.absorbed_by_sleep else 0,
                    self._serialize_json(node.Q),
                    self._serialize_json(node.n),
                    self._serialize_json(node.Q_omega),
                    self._serialize_json(node.n_omega),
                    self._serialize_json(node.secondary_parents),
                    self._serialize_json(node.evidence_ids),
                ),
            )

    def _fetch_representation(self, node_id: str) -> SkillRepresentation:
        row = self._db.execute(
            "SELECT node_id, content, embedding FROM skill_representation WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            raise KeyError(node_id)
        return SkillRepresentation(
            id=row["node_id"],
            content=row["content"],
            embedding=self._deserialize_embedding(row["embedding"]),
        )

    @staticmethod
    def _deserialize_json(blob: Any, default: Any) -> Any:
        if blob is None:
            return default
        if isinstance(blob, bytes):
            text = blob.decode("utf-8")
        else:
            text = str(blob)
        try:
            return json.loads(text)
        except Exception:
            return default

    def _fetch_graph_state(self, node_id: str) -> SkillNode:
        row = self._db.execute(
            """
            SELECT node_id, depth, parent_id, task_type_primary, t_create,
                   last_accessed_step, decay_rate, absorbed_by_sleep, Q, n,
                   Q_omega, n_omega, secondary_parents, evidence_ids
            FROM skill_graph_state
            WHERE node_id = ?
            """,
            (node_id,),
        ).fetchone()
        if row is None:
            raise KeyError(node_id)
        return SkillNode(
            id=row["node_id"],
            task_type_primary=row["task_type_primary"],
            t_create=int(row["t_create"]),
            depth=int(row["depth"]),
            parent_id=row["parent_id"],
            secondary_parents=list(self._deserialize_json(row["secondary_parents"], [])),
            last_accessed_step=int(row["last_accessed_step"]),
            Q=dict(self._deserialize_json(row["Q"], {})),
            n=dict(self._deserialize_json(row["n"], {})),
            Q_omega=dict(self._deserialize_json(row["Q_omega"], {})),
            n_omega=dict(self._deserialize_json(row["n_omega"], {})),
            decay_rate=float(row["decay_rate"]),
            evidence_ids=list(self._deserialize_json(row["evidence_ids"], [])),
            absorbed_by_sleep=bool(row["absorbed_by_sleep"]),
        )

    def _load_graph_state(self) -> None:
        rows = self._db.execute(
            """
            SELECT node_id
            FROM skill_graph_state
            ORDER BY depth ASC, t_create ASC, node_id ASC
            """
        ).fetchall()
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
        task_type_primary: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SkillRepresentation]:
        if node_ids is None:
            clauses: List[str] = []
            params: List[Any] = []
            if depth is not None:
                clauses.append("depth = ?")
                params.append(depth)
            if task_type_primary is not None:
                clauses.append("task_type_primary = ?")
                params.append(task_type_primary)

            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""
            query = (
                "SELECT node_id FROM skill_graph_state "
                f"{where} "
                "ORDER BY depth ASC, t_create ASC, node_id ASC"
                f"{limit_sql}"
            )
            rows = self._db.execute(query, tuple(params)).fetchall()
            node_ids = [row["node_id"] for row in rows]
        if not node_ids:
            return []
        placeholders = ",".join("?" for _ in node_ids)
        query = (
            "SELECT node_id, content, embedding FROM skill_representation "
            f"WHERE node_id IN ({placeholders})"
        )
        rows = self._db.execute(query, tuple(node_ids)).fetchall()
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

    @property
    def lambda_slow(self) -> Optional[float]:
        return self.memory_config.lambda_slow

    @property
    def lambda_fast(self) -> Optional[float]:
        return self.memory_config.effective_lambda_fast

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
        task_type_primary: str,
        t_create: int,
        depth: int,
        parent_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        evidence_ids: Optional[List[str]] = None,
    ) -> SkillNode:
        """Create a node from text and add it to the service."""
        if embedding is None:
            if self.embedding_provider is None:
                raise ValueError(
                    "embedding_provider is required when embedding is omitted"
                )
            embedding = self.embedding_provider.embed_single(content)

        representation = SkillRepresentation(
            id=id,
            content=content,
            embedding=embedding,
        )

        node = SkillNode(
            id=id,
            task_type_primary=task_type_primary,
            t_create=t_create,
            depth=depth,
            parent_id=parent_id,
            evidence_ids=list(evidence_ids or []),
        )
        return self.add_node(node, representation, parent_id=parent_id)

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
        task_type_primary: Optional[str] = None,
    ) -> List[SkillRepresentation]:
        return self._fetch_representations(
            depth=depth,
            task_type_primary=task_type_primary,
        )

    def search_nodes(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 5,
        depth: Optional[int] = None,
        task_type_primary: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Similarity search over stored skill nodes."""
        return self.retriever.search(
            self._fetch_representations(depth=depth, task_type_primary=task_type_primary),
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
        task_type_primary: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Alias for embedding-based similarity search."""
        return self.search_nodes(
            query_embedding,
            top_k=top_k,
            depth=depth,
            task_type_primary=task_type_primary,
        )

    def search_by_text(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        depth: Optional[int] = None,
        task_type_primary: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convenience wrapper that embeds text before similarity search."""
        if self.embedding_provider is None:
            raise ValueError("embedding_provider is required for text search")
        query_embedding = self.embedding_provider.embed_single(query_text)
        return self.search_nodes(
            query_embedding,
            top_k=top_k,
            depth=depth,
            task_type_primary=task_type_primary,
        )

    def find_best_parent(
        self,
        query_embedding: List[float],
        *,
        target_depth: int,
        task_type_primary: Optional[str] = None,
    ) -> Optional[SkillNode]:
        """Find the best parent candidate for a query embedding."""
        best = self.retriever.best_node(
            self._fetch_representations(
                depth=target_depth,
                task_type_primary=task_type_primary,
            ),
            query_embedding,
            depth=None,
        )
        if best is None:
            return None
        return self.graph.get(best.id)

    def close(self) -> None:
        """Close the SQLite connection."""
        self._db.close()

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its subtree from the graph and SQLite table."""
        removed_ids = self.graph.remove(node_id)
        with self._db:
            for rid in removed_ids:
                self._db.execute(
                    "DELETE FROM skill_representation WHERE node_id = ?",
                    (rid,),
                )
                self._db.execute(
                    "DELETE FROM skill_graph_state WHERE node_id = ?",
                    (rid,),
                )

    def refresh_content_db(self) -> None:
        """Synchronize both SQLite tables with the graph."""
        valid_ids = list(self.graph.nodes)
        if not valid_ids:
            with self._db:
                self._db.execute("DELETE FROM skill_representation")
                self._db.execute("DELETE FROM skill_graph_state")
            return

        placeholders = ",".join("?" for _ in valid_ids)
        with self._db:
            self._db.execute(
                f"""
                DELETE FROM skill_representation
                WHERE node_id NOT IN ({placeholders})
                """,
                tuple(valid_ids),
            )
            self._db.execute(
                f"""
                DELETE FROM skill_graph_state
                WHERE node_id NOT IN ({placeholders})
                """,
                tuple(valid_ids),
            )

    def retrieve_query(
        self,
        query_text: str,
        *,
        k: int = 5,
        depth: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Compatibility alias for text-based similarity search."""
        return self.search_by_text(query_text, top_k=k, depth=depth)
