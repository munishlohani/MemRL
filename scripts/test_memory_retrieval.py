#!/usr/bin/env python3
"""Smoke test for state-aware memory retrieval and strategic cold-task fallback."""

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
        text = text.lower()
        if "strategy" in text or "cold" in text:
            return [0.0, 1.0]
        if "fresh" in text:
            return [0.2, 1.0]
        return [1.0, 0.0]


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "skill_memory.sqlite")
        config = SimpleNamespace(
            lambda_base=0.1,
            lambda_shrink=10.0,
            epsilon_decay=0.01,
            skill_db_path=db_path,
        )
        service = MemoryService(config, embedding_provider=StubEmbedder())

        old = service.add_node_from_text(
            id="tactical-old",
            content="old tactical memory",
            task_type_dominant="alfworld",
            t_create=1,
            depth=2,
            embedding=[1.0, 0.0],
        )
        fresh = service.add_node_from_text(
            id="tactical-fresh",
            content="fresh tactical memory",
            task_type_dominant="alfworld",
            t_create=2,
            depth=2,
            embedding=[0.8, 0.2],
        )

        old_node = service.get_node(old.id)
        old_node.Q["alfworld"] = 0.1
        old_node.n["alfworld"] = 1
        old_node.last_accessed_step = 0
        service.graph.refresh_decay_rate(old_node)

        fresh_node = service.get_node(fresh.id)
        fresh_node.Q["alfworld"] = 2.0
        fresh_node.n["alfworld"] = 20
        fresh_node.last_accessed_step = 9
        service.graph.refresh_decay_rate(fresh_node)

        service.graph.current_step = 10
        tactical_result = service.retrieve_query(
            "tool use",
            k=1,
            threshold=0.0,
        )
        tactical_payload, tactical_queries = tactical_result
        tactical_selected = tactical_payload["selected"]
        assert tactical_selected and tactical_selected[0]["memory_id"] == fresh.id
        assert tactical_payload["simmax"] == tactical_selected[0]["similarity"]
        assert tactical_queries == [("tool use", 1.0)]

        strat_a = service.add_node_from_text(
            id="strategic-a",
            content="strategy alpha",
            task_type_dominant="train",
            t_create=3,
            depth=1,
            embedding=[0.0, 1.0],
        )
        strat_b = service.add_node_from_text(
            id="strategic-b",
            content="strategy beta",
            task_type_dominant="train",
            t_create=4,
            depth=1,
            embedding=[0.0, 1.0],
        )

        strat_a_node = service.get_node(strat_a.id)
        strat_a_node.Q_omega = {"train": 10.0, "eval": 0.0}
        strat_a_node.n_omega = {"train": 100, "eval": 1}
        strat_b_node = service.get_node(strat_b.id)
        strat_b_node.Q_omega = {"train": 6.0, "eval": 6.0}
        strat_b_node.n_omega = {"train": 1, "eval": 1}

        strategic_result = service.retrieve_query(
            "cold task strategy",
            k=1,
            depth=1,
            task_type_dominant="cold_task",
        )
        strategic_payload, strategic_queries = strategic_result
        strategic_selected = strategic_payload["selected"]
        assert strategic_selected and strategic_selected[0]["memory_id"] == strat_a.id
        assert round(strategic_selected[0]["q_estimate"], 5) == round(
            (100 / 101) * 10.0 + (1 / 101) * 0.0,
            5,
        )
        assert strategic_queries == [("cold_task", 1.0)]

        service.close()
        print("retrieval-ok", db_path)


if __name__ == "__main__":
    main()
