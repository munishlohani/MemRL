#!/usr/bin/env python3
"""Smoke test for memory graph ingestion and SQLite representation writes."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memrl.service.memory_service import MemoryService


class StubEmbedder:
    def embed_single(self, text: str) -> list[float]:
        return [float(len(text)), float(sum(ord(c) for c in text) % 10)]


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "skill_memory.sqlite")
        config = SimpleNamespace(
            lambda_slow=0.1,
            effective_lambda_fast=0.5,
            epsilon_decay=0.01,
            skill_db_path=db_path,
        )
        service = MemoryService(config, embedding_provider=StubEmbedder())

        parent = service.add_node_from_text(
            id="node-parent",
            content="collect tools",
            task_type_primary="alfworld",
            t_create=1,
            depth=3,
            embedding=[1.0, 2.0],
        )
        child = service.add_node_from_text(
            id="node-child",
            content="open drawer",
            task_type_primary="alfworld",
            t_create=2,
            depth=3,
            parent_id=parent.id,
            embedding=[2.0, 3.0],
        )

        assert service.get_node(parent.id).id == parent.id
        assert service.get_node(child.id).parent_id == parent.id
        assert service.graph.child_ids(parent.id) == {child.id}

        parent_rep = service.get_representation(parent.id)
        child_rep = service.get_representation(child.id)
        assert parent_rep.content == "collect tools"
        assert child_rep.content == "open drawer"
        assert child_rep.embedding == [2.0, 3.0]

        hits = service.search([2.0, 3.0], top_k=1)
        assert hits and hits[0]["node_id"] == child.id

        service.close()
        print("ingestion-ok", db_path)


if __name__ == "__main__":
    main()
