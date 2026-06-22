"""Skill graph primitives for the updated MemRL design."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

from .skill_node import SkillNode


@dataclass
class SkillGraph:
    """Unified hierarchical graph for tactical skills and strategic scaffolds."""

    lambda_slow: Optional[float] = None
    lambda_fast: Optional[float] = None
    lambda_shrink: float = 10.0
    epsilon: float = 0.01
    root_id: str = "root"
    current_step: int = 0
    nodes: Dict[str, SkillNode] = field(default_factory=dict)

    @property
    def lambda_d(self) -> Dict[int, float]:
        """Depth-indexed base decay rates."""
        slow = float(self.lambda_slow) if self.lambda_slow is not None else 0.0
        fast = (
            float(self.lambda_fast)
            if self.lambda_fast is not None
            else 5.0 * slow
        )
        return {1: 0.0, 2: slow, 3: fast}

    def lambda_for_depth(self, depth: int) -> float:
        return self.lambda_d.get(depth, 0.0)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def get(self, node_id: str) -> SkillNode:
        return self.nodes[node_id]

    def nodes_at_depth(self, depth: int) -> List[SkillNode]:
        return [node for node in self.nodes.values() if node.depth == depth]

    def node_count(self, depth: Optional[int] = None) -> int:
        """Count nodes globally or within a specific depth."""
        if depth is None:
            return len(self.nodes)
        return sum(1 for node in self.nodes.values() if node.depth == depth)

    def unabsorbed_tactical_count(self) -> int:
        """Count tactical d=2 nodes that have not yet been absorbed by sleep."""
        return sum(
            1
            for node in self.nodes.values()
            if node.depth == 2 and not node.absorbed_by_sleep
        )

    def child_ids(self, parent_id: str) -> Set[str]:
        return {
            node.id
            for node in self.nodes.values()
            if (node.parent_id or self.root_id) == parent_id
        }

    def insert(self, node: SkillNode, parent_id: Optional[str] = None) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node already exists: {node.id}")

        resolved_parent = parent_id if parent_id is not None else node.parent_id
        if resolved_parent is None:
            resolved_parent = self.root_id
        if resolved_parent != self.root_id and resolved_parent not in self.nodes:
            raise KeyError(f"Parent node not found: {resolved_parent}")

        self.nodes[node.id] = node
        node.parent_id = resolved_parent
        self.refresh_decay_rate(node)

    def refresh_decay_rate(self, node: SkillNode) -> None:
        """Update the cached decay rate for a node."""
        node.recompute_decay_rate(
            lambda_d=self.lambda_for_depth(node.depth),
            epsilon=self.epsilon,
            lambda_shrink=self.lambda_shrink,
        )

    def reparent(self, node_or_id: Union[SkillNode, str], new_parent_id: str) -> None:
        node = self._resolve_node(node_or_id)
        if new_parent_id != self.root_id and new_parent_id not in self.nodes:
            raise KeyError(f"Parent node not found: {new_parent_id}")

        old_parent_id = node.parent_id or self.root_id
        if old_parent_id == new_parent_id:
            return

        node.parent_id = new_parent_id

    def remove(self, node_or_id: Union[SkillNode, str]) -> List[str]:
        node = self._resolve_node(node_or_id)
        removed: List[str] = []
        self._remove_subtree(node.id, removed)
        return removed

    def _resolve_node(self, node_or_id: Union[SkillNode, str]) -> SkillNode:
        if isinstance(node_or_id, SkillNode):
            return node_or_id
        return self.nodes[node_or_id]

    def _remove_subtree(self, node_id: str, removed: List[str]) -> None:
        for child_id in list(self.child_ids(node_id)):
            self._remove_subtree(child_id, removed)

        self.nodes.pop(node_id, None)
        removed.append(node_id)
