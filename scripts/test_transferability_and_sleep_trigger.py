#!/usr/bin/env python3
"""Smoke test for transferability scoring and sleep trigger counts."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memrl.service.memory_service import MemoryService
from memrl.memory.skill_node import SkillNode
from memrl.memory.skill_representation import SkillRepresentation


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
            lambda_shrink=10.0,
            theta_1=0.75,
            n_min=8,
            theta_cv=0.5,
            n_sleep=2,
            skill_db_path=db_path,
        )
        service = MemoryService(config, embedding_provider=StubEmbedder())

        strong = service.add_node_from_text(
            id="node-strong",
            content="stack objects",
            task_type_primary="alfworld",
            t_create=1,
            depth=3,
            embedding=[1.0, 0.0],
        )
        strong.Q = {"task-a": 0.8, "task-b": 0.2}
        strong.n = {"task-a": 10, "task-b": 2}
        service._upsert_graph_state(strong)

        weak = service.add_node_from_text(
            id="node-weak",
            content="open drawer",
            task_type_primary="alfworld",
            t_create=2,
            depth=3,
            embedding=[0.0, 1.0],
        )
        weak.Q = {"task-a": 0.1, "task-b": 0.9}
        weak.n = {"task-a": 4, "task-b": 4}
        service._upsert_graph_state(weak)

        strong_score = service.transferability_score(strong.id)
        weak_score = service.transferability_score(weak.id)

        assert math.isclose(strong_score, 0.8622448979591837, rel_tol=1e-9)
        assert weak_score < config.theta_1
        assert service.should_float_up(strong.id) is True
        assert service.should_float_up(weak.id) is False

        sleepy_one = SkillNode(
            id="node-sleep-1",
            task_type_primary="alfworld",
            t_create=3,
            depth=2,
            parent_id=None,
        )
        sleepy_two = SkillNode(
            id="node-sleep-2",
            task_type_primary="alfworld",
            t_create=4,
            depth=2,
            parent_id=None,
        )
        service.add_node(
            sleepy_one,
            SkillRepresentation(
                id=sleepy_one.id,
                content="sleep one",
                embedding=[0.2, 0.2],
            ),
        )
        service.add_node(
            sleepy_two,
            SkillRepresentation(
                id=sleepy_two.id,
                content="sleep two",
                embedding=[0.3, 0.3],
            ),
        )

        assert service.sleep_consolidation_count() == 2
        assert service.should_sleep_consolidate() is True

        service.close()
        print("transferability-and-sleep-ok", db_path)


if __name__ == "__main__":
    main()
