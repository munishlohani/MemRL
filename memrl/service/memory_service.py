"""Research-native memory service for the updated MemRL design."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
try:
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

    _HAS_SQLALCHEMY = True
except Exception:  # pragma: no cover - optional runtime dependency
    Boolean = Column = Float = Integer = JSON = LargeBinary = MetaData = String = Table = Text = Any  # type: ignore[assignment]
    create_engine = delete = select = sqlite_insert = None  # type: ignore[assignment]
    _HAS_SQLALCHEMY = False

from ..memory.graph import SkillGraph
from ..memory.skill_node import SkillNode
from ..memory.skill_representation import SkillRepresentation
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
            lambda_slow=memory_config.lambda_slow,
            lambda_fast=memory_config.effective_lambda_fast,
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

        self._use_sqlalchemy = _HAS_SQLALCHEMY
        self._db = None
        self._engine = None
        self._metadata = None
        self.skill_representation_table = None
        self.skill_graph_state_table = None

        if self._use_sqlalchemy:
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
                Column("task_type_primary", String, nullable=False),
                Column("t_create", Integer, nullable=False),
                Column("last_accessed_step", Integer, nullable=False),
                Column("decay_rate", Float, nullable=False),
                Column("absorbed_by_sleep", Boolean, nullable=False, default=False),
                Column("Q", JSON, nullable=False),
                Column("n", JSON, nullable=False),
                Column("Q_omega", JSON, nullable=False),
                Column("n_omega", JSON, nullable=False),
                Column("secondary_parents", JSON, nullable=False),
                Column("evidence_ids", JSON, nullable=False),
            )
            self._metadata.create_all(self._engine)
        else:
            self._db = sqlite3.connect(self.db_path)
            self._db.row_factory = sqlite3.Row
            self._init_sqlite_schema()

        if self.graph.nodes:
            self._sync_graph_state_to_db()
        else:
            self._load_graph_state()
        self.retriever = SkillSimilarityRetriever()

    @staticmethod
    def _serialize_embedding(embedding: List[float]) -> bytes:
        array = np.asarray([float(value) for value in embedding], dtype=np.float32)
        return array.tobytes()

    @staticmethod
    def _deserialize_embedding(blob: Any) -> List[float]:
        if blob is None:
            return []
        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        if isinstance(blob, (bytes, bytearray)):
            if len(blob) % np.dtype(np.float32).itemsize == 0:
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
            task_type_primary=node.task_type_primary,
            t_create=node.t_create,
            last_accessed_step=node.last_accessed_step,
            decay_rate=node.decay_rate,
            absorbed_by_sleep=bool(node.absorbed_by_sleep),
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
                "task_type_primary": stmt.excluded.task_type_primary,
                "t_create": stmt.excluded.t_create,
                "last_accessed_step": stmt.excluded.last_accessed_step,
                "decay_rate": stmt.excluded.decay_rate,
                "absorbed_by_sleep": stmt.excluded.absorbed_by_sleep,
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
            task_type_primary=row["task_type_primary"],
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
            absorbed_by_sleep=bool(row["absorbed_by_sleep"]),
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
        task_type_primary: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SkillRepresentation]:
        if node_ids is None:
            stmt = select(self.skill_graph_state_table.c.node_id)
            if depth is not None:
                stmt = stmt.where(self.skill_graph_state_table.c.depth == depth)
            if task_type_primary is not None:
                stmt = stmt.where(
                    self.skill_graph_state_table.c.task_type_primary == task_type_primary
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

        if depth == 1:
            node = SkillNode.create_strategic(
                id=id,
                task_type_primary=task_type_primary,
                t_create=t_create,
                parent_id=parent_id,
                evidence_ids=list(evidence_ids or []),
            )
        elif depth == 3:
            node = SkillNode.create_tactical(
                id=id,
                task_type_primary=task_type_primary,
                t_create=t_create,
                parent_id=parent_id,
                evidence_ids=list(evidence_ids or []),
            )
        else:
            raise ValueError("depth must be 1 or 3")

        representation = SkillRepresentation(
            id=id,
            content=content,
            embedding=embedding,
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
            self._fetch_representations(
                depth=depth,
                task_type_primary=task_type_primary,
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
    ) -> List[Dict[str, Any]]:
        """Compatibility alias for text-based similarity search."""
        return self.search_by_text(query_text, top_k=k, depth=depth)
