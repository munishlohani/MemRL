#!/usr/bin/env python3
"""Smoke test for reloading a persisted memory graph from SQLite."""

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
            lambda_base=0.1,
            epsilon_decay=0.01,
            skill_db_path=db_path,
        )

        first = MemoryService(config, embedding_provider=StubEmbedder())
        first.add_node_from_text(
            id="node-root",
            content="scan room",
            task_type_dominant="alfworld",
            t_create=1,
            depth=2,
            embedding=[1.0, 1.0],
        )
        first.add_node_from_text(
            id="node-leaf",
            content="pick apple",
            task_type_dominant="alfworld",
            t_create=2,
            depth=2,
            parent_id="node-root",
            embedding=[4.0, 5.0],
        )
        first.close()

        second = MemoryService(config, embedding_provider=StubEmbedder())
        root = second.get_node("node-root")
        leaf = second.get_node("node-leaf")
        rep = second.get_representation("node-leaf")
        hits = second.search([4.0, 5.0], top_k=1)

        assert root.id == "node-root"
        assert leaf.parent_id == "node-root"
        assert rep.content == "pick apple"
        assert rep.embedding == [4.0, 5.0]
        assert hits and hits[0]["node_id"] == "node-leaf"

        second.close()
        print("persistence-ok", db_path)


if __name__ == "__main__":
    main()
