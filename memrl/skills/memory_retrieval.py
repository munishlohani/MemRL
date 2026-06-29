"""Memory retrieval skill shared by agents and episode runners."""

from __future__ import annotations

import copy
from functools import lru_cache
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from memrl.service.memory_service import MemoryService


_SKILL_DOC_PATH = Path(__file__).resolve().parent / "memory_retrieval_skill" / "SKILL.md"


@lru_cache(maxsize=1)
def _load_skill_contract_text() -> str:
    """Load the retrieval skill contract from disk, with a short fallback."""
    try:
        return _SKILL_DOC_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return (
            "Memory retrieval skill contract unavailable.\n"
            "Treat retrieved memories as advisory context only and still answer in "
            "the normal Thought/Action format."
        )


@dataclass
class MemoryRetrievalResult:
    """Structured output of a memory retrieval skill call."""

    query_text: str
    context_text: str
    selected_memories: List[Dict[str, Any]] = field(default_factory=list)
    topk_queries: List[Tuple[str, float]] = field(default_factory=list)
    task_description: str = ""
    observation: str = ""
    task_type: str = "unknown"
    episode_id: str = "unknown"
    history_text: str = ""
    active_strategic_node_id: Optional[str] = None
    current_step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result into a plain dictionary for logging or JSON."""
        payload = asdict(self)
        payload["selected_ids"] = [
            str(item.get("memory_id") or item.get("id") or "")
            for item in self.selected_memories
            if isinstance(item, dict)
        ]
        payload["selected_count"] = len(self.selected_memories)
        return payload

    def to_tool_message(self, skill_name: str = "memory_retrieval") -> Dict[str, str]:
        """Render the result as a tool message for the agent conversation."""
        content = self.context_text.strip() or "No archived memories."
        return {
            "role": "tool",
            "name": skill_name,
            "content": f"Memory retrieval result:\n{content}",
        }


class MemoryRetrievalSkill:
    """Build a retrieval query and materialize a prompt-ready memory context."""

    def __init__(
        self,
        *,
        memory_service: MemoryService,
        retrieve_k: int = 1,
        rl_config: Optional[Any] = None,
    ) -> None:
        self.memory_service = memory_service
        self.retrieve_k = max(1, int(retrieve_k))
        self.rl_config = rl_config
        self._skill_contract_text = _load_skill_contract_text()

    def prompt_contract(self) -> str:
        """Return the contract text injected into the agent prompt."""
        return self._skill_contract_text

    def retrieve(
        self,
        *,
        task_description: str,
        observation: str,
        history_messages: Sequence[Dict[str, str]],
        task_type: str,
        episode_id: str,
        active_strategic_node_id: Optional[str] = None,
        current_step: Optional[int] = None,
        query_override: Optional[str] = None,
    ) -> MemoryRetrievalResult:
        """Run retrieval and return both context text and structured metadata."""
        query_text = self.build_query(
            task_description=task_description,
            observation=observation,
            history_messages=history_messages,
        )
        if query_override is not None and str(query_override).strip():
            query_text = "\n".join(
                [
                    query_text,
                    f"Skill query: {str(query_override).strip()}",
                ]
            )
        history_text = self._history_messages_to_text(history_messages)
        retrieved_memories, topk_queries = self._retrieve(
            query_text=query_text,
            task_type=task_type,
            active_strategic_node_id=active_strategic_node_id,
            current_step=current_step,
        )
        context_text = self.format_selected_memories(retrieved_memories)
        return MemoryRetrievalResult(
            query_text=query_text,
            context_text=context_text,
            selected_memories=copy.deepcopy(retrieved_memories),
            topk_queries=list(topk_queries or []),
            task_description=task_description,
            observation=observation,
            task_type=task_type,
            episode_id=episode_id,
            history_text=history_text,
            active_strategic_node_id=active_strategic_node_id,
            current_step=current_step,
        )

    def build_query(
        self,
        *,
        task_description: str,
        observation: str,
        history_messages: Sequence[Dict[str, str]],
    ) -> str:
        """Create the retrieval query text used for memory search."""
        history_text = self._history_messages_to_text(history_messages)
        parts = [
            f"Task: {task_description.strip()}",
            f"Observation: {observation.strip()}",
        ]
        if history_text:
            parts.append(f"History: {history_text}")
        return "\n".join(part for part in parts if part.strip())

    def format_selected_memories(self, selected: Sequence[Dict[str, Any]]) -> str:
        """Format retrieved memories into an agent prompt block."""
        if not selected:
            return "No archived memories."

        parts: List[str] = []
        for idx, item in enumerate(selected, 1):
            if not isinstance(item, dict):
                continue
            mem_id = str(item.get("memory_id") or item.get("id") or f"memory-{idx}")
            content = str(item.get("content") or "").strip()
            score = float(item.get("score", 0.0) or 0.0)
            if not content:
                continue
            parts.append(f"{idx}. [{mem_id}] (score={score:.3f}) {content}")

        return "\n".join(parts) if parts else "No archived memories."

    def _retrieve(
        self,
        *,
        query_text: str,
        task_type: str,
        active_strategic_node_id: Optional[str],
        current_step: Optional[int],
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, float]]]:
        try:
            tau = float(
                getattr(self.rl_config, "sim_threshold", getattr(self.rl_config, "tau", 0.0))
            )
        except Exception:
            tau = 0.0

        try:
            result, topk_queries = self.memory_service.retrieve_query(
                query_text,
                k=self.retrieve_k,
                threshold=tau,
                task_type_dominant=task_type,
                active_strategic_node_id=active_strategic_node_id,
                current_step=current_step,
            )
        except Exception:
            return [], []

        selected = (result or {}).get("selected", [])
        if not isinstance(selected, list):
            selected = []
        return selected, list(topk_queries or [])

    @staticmethod
    def _history_messages_to_text(history_messages: Sequence[Dict[str, str]]) -> str:
        lines: List[str] = []
        for message in history_messages[-10:]:
            role = str(message.get("role", "user")).strip()
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)


__all__ = [
    "MemoryRetrievalResult",
    "MemoryRetrievalSkill",
]
