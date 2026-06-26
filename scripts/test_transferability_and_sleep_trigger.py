#!/usr/bin/env python3
"""Smoke test for sleep trigger counts."""

from __future__ import annotations

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
            lambda_base=0.1,
            epsilon_decay=0.01,
            lambda_shrink=10.0,
            n_sleep=2,
            skill_db_path=db_path,
        )
        service = MemoryService(config, embedding_provider=StubEmbedder())

        sleepy_one = SkillNode(
            id="node-sleep-1",
            task_type_dominant="alfworld",
            t_create=3,
            depth=2,
            parent_id=None,
        )
        sleepy_two = SkillNode(
            id="node-sleep-2",
            task_type_dominant="alfworld",
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
