"""LLM-backed tactical formation judgment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..providers.base import BaseLLM


@dataclass
class TacticalFormationCandidate:
    """One positive-TD experience awaiting formation judgment."""

    candidate_id: str
    task_type: str
    task_description: str
    episode_id: str
    episode_index: int
    step_index: int
    observation: str
    action: str
    reward: float
    td_error: float
    history: str
    retrieved_memories: str
    retrieved_ids: List[str] = field(default_factory=list)
    source_memory_id: Optional[str] = None
    active_strategic_node_id: Optional[str] = None

    def render(self) -> str:
        """Render the candidate as prompt-ready text."""
        fields = [
            f"candidate_id: {self.candidate_id}",
            f"task_type: {self.task_type}",
            f"task_description: {self.task_description}",
            f"episode_id: {self.episode_id}",
            f"episode_index: {self.episode_index}",
            f"step_index: {self.step_index}",
            f"observation: {self.observation}",
            f"action: {self.action}",
            f"reward: {self.reward:.6f}",
            f"td_error: {self.td_error:.6f}",
            f"source_memory_id: {self.source_memory_id or 'none'}",
            f"active_strategic_node_id: {self.active_strategic_node_id or 'none'}",
            f"retrieved_ids: {', '.join(self.retrieved_ids) if self.retrieved_ids else 'none'}",
            f"retrieved_memories: {self.retrieved_memories or 'No archived memories.'}",
            f"history: {self.history or 'No prior trajectory.'}",
        ]
        return "\n".join(fields)

    def fallback_summary(self) -> str:
        """Create a deterministic summary if the judge omits one."""
        return (
            f"{self.task_type}: {self.action.strip()} after {self.observation.strip()}"
            f" (reward={self.reward:.3f}, td={self.td_error:.3f})"
        ).strip()


@dataclass
class TacticalFormationDecision:
    """Decision returned by the tactical formation judge."""

    candidate_id: str
    approved: bool
    summary: Optional[str] = None


@dataclass
class TacticalSummaryDraft:
    """Structured tactical summary used for storage and embedding."""

    title: str
    goal: str
    setup: str
    procedure: List[str] = field(default_factory=list)
    outcome: str = ""
    reusable_rule: str = ""
    failure_modes: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Render the draft into a canonical summary string."""
        lines: List[str] = []
        if self.title.strip():
            lines.append(f"TITLE: {self.title.strip()}")
        if self.goal.strip():
            lines.append(f"GOAL: {self.goal.strip()}")
        if self.setup.strip():
            lines.append(f"SETUP: {self.setup.strip()}")
        if self.procedure:
            lines.append("PROCEDURE:")
            for idx, step in enumerate(self.procedure, 1):
                step_text = str(step).strip()
                if step_text:
                    lines.append(f"{idx}. {step_text}")
        if self.outcome.strip():
            lines.append(f"OUTCOME: {self.outcome.strip()}")
        if self.reusable_rule.strip():
            lines.append(f"REUSABLE RULE: {self.reusable_rule.strip()}")
        if self.failure_modes:
            lines.append("FAILURE MODES:")
            for idx, item in enumerate(self.failure_modes, 1):
                item_text = str(item).strip()
                if item_text:
                    lines.append(f"{idx}. {item_text}")
        return "\n".join(lines).strip()


TACTICAL_SUMMARY_PROMPT = """You are converting one successful positive-TD experience into a reusable tactical memory summary.

The summary will be embedded and retrieved later, so make it semantically informative rather than a noisy trajectory dump.
Only describe the successful path that worked; failed attempts are filtered upstream and will not be stored.

Return a single JSON object and nothing else.

Schema:
{{
  "title": string,
  "goal": string,
  "setup": string,
  "procedure": [string, ...],
  "outcome": string,
  "reusable_rule": string,
  "failure_modes": [string, ...]
}}

Rules:
- Write for reuse, not narration.
- Compress the experience into a clear procedural pattern.
- Do not copy the raw step-by-step trace.
- Keep the summary short, specific, and action-oriented.
- procedure should contain 2 to 6 high-signal steps.
- failure_modes should list when the pattern should not be used or what can go wrong.
- Omit incidental observations unless they matter for reuse.
- Do not include markdown, explanations, or extra keys.

Source experience:
{candidate}
"""


TACTICAL_FORMATION_PROMPT = """You are deciding whether each positive-TD experience should become a tactical memory.

Return a single JSON object and nothing else.

Schema:
{
  "decisions": [
    {
      "candidate_id": string,
      "approved": boolean,
      "summary": string | null
    }
  ]
}

Rules:
- Only judge the candidates listed below.
- Only approve successful, reusable experiences; failed attempts are filtered upstream and should be rejected if they slip through.
- Approve only if the experience is reusable, semantically coherent, and not just a one-off outcome.
- Reject failed, noisy, duplicated, or environment-specific behavior.
- Every approved item must include a concise reusable summary suitable for memory storage.
- Every rejected item must set summary to null.
- Do not add explanations, markdown, extra keys, or nested structures.
- Preserve the candidate_id for each decision.

Episode context:
{episode_context}

Candidates:
{candidates}
"""


class TacticalFormationJudge:
    """LLM-backed batch judge for tactical memory formation."""

    def __init__(self, llm_provider: BaseLLM) -> None:
        self.llm_provider = llm_provider

    def judge_candidates(
        self,
        candidates: Sequence[TacticalFormationCandidate],
    ) -> List[TacticalFormationDecision]:
        if not candidates:
            return []

        prompt = TACTICAL_FORMATION_PROMPT.format(
            episode_context=self._render_episode_context(candidates),
            candidates=self._render_candidates(candidates),
        )
        response = self.llm_provider.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._parse_response(response, candidates)

    def _render_episode_context(
        self,
        candidates: Sequence[TacticalFormationCandidate],
    ) -> str:
        first = candidates[0]
        lines = [
            f"task_type: {first.task_type}",
            f"task_description: {first.task_description}",
            f"episode_id: {first.episode_id}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_candidates(candidates: Sequence[TacticalFormationCandidate]) -> str:
        return "\n\n".join(
            f"[{idx + 1}]\n{candidate.render()}" for idx, candidate in enumerate(candidates)
        )

    def _parse_response(
        self,
        response: str,
        candidates: Sequence[TacticalFormationCandidate],
    ) -> List[TacticalFormationDecision]:
        payload = self._load_json_object(response)

        if "decisions" in payload:
            raw_decisions = payload.get("decisions", [])
        else:
            raw_decisions = [payload]

        if not isinstance(raw_decisions, list):
            raise ValueError(f"Formation judge response must contain a list of decisions: {response!r}")

        by_candidate_id = {candidate.candidate_id: candidate for candidate in candidates}
        decisions: List[TacticalFormationDecision] = []
        for raw in raw_decisions:
            if not isinstance(raw, dict):
                continue
            candidate_id = self._coerce_optional_str(raw.get("candidate_id"))
            if candidate_id is None:
                continue
            if candidate_id not in by_candidate_id:
                continue

            approved = self._coerce_bool(
                raw.get("approved", raw.get("store", False))
            )
            summary = self._coerce_optional_str(raw.get("summary"))
            if approved and summary is None:
                summary = by_candidate_id[candidate_id].fallback_summary()
            decisions.append(
                TacticalFormationDecision(
                    candidate_id=candidate_id,
                    approved=approved,
                    summary=summary if approved else None,
                )
            )

        seen = {decision.candidate_id for decision in decisions}
        for candidate in candidates:
            if candidate.candidate_id in seen:
                continue
            decisions.append(
                TacticalFormationDecision(
                    candidate_id=candidate.candidate_id,
                    approved=False,
                    summary=None,
                )
            )

        return decisions

    @staticmethod
    def _load_json_object(response: str) -> Dict[str, Any]:
        text = response.strip()
        if not text:
            raise ValueError("Empty formation judge response")

        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"Unable to parse formation judge response: {response!r}")
            loaded = json.loads(text[start : end + 1])

        if not isinstance(loaded, dict):
            raise ValueError(f"Formation judge response must be a JSON object: {response!r}")
        return loaded

    @staticmethod
    def _coerce_optional_str(value: object) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _coerce_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            return lowered in {"true", "yes", "1", "approve", "approved"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False


class TacticalSummaryWriter:
    """LLM-backed structured summary writer for tactical memories."""

    def __init__(self, llm_provider: BaseLLM) -> None:
        self.llm_provider = llm_provider

    def summarize_candidate(self, candidate: TacticalFormationCandidate) -> TacticalSummaryDraft:
        if candidate is None:
            raise ValueError("candidate is required")

        prompt = TACTICAL_SUMMARY_PROMPT.format(candidate=candidate.render())
        response = self.llm_provider.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._parse_response(response, candidate)

    @staticmethod
    def format_summary(draft: TacticalSummaryDraft) -> str:
        return draft.to_text()

    def _parse_response(
        self,
        response: str,
        candidate: TacticalFormationCandidate,
    ) -> TacticalSummaryDraft:
        payload = self._load_json_object(response)

        title = self._coerce_optional_str(payload.get("title")) or candidate.task_description
        goal = self._coerce_optional_str(payload.get("goal")) or candidate.task_description
        setup = self._coerce_optional_str(payload.get("setup")) or candidate.observation
        procedure = self._coerce_str_list(payload.get("procedure"))
        if not procedure:
            procedure = [candidate.fallback_summary()]
        outcome = self._coerce_optional_str(payload.get("outcome")) or f"Reward={candidate.reward:.3f}"
        reusable_rule = self._coerce_optional_str(payload.get("reusable_rule")) or candidate.fallback_summary()
        failure_modes = self._coerce_str_list(payload.get("failure_modes"))

        return TacticalSummaryDraft(
            title=title,
            goal=goal,
            setup=setup,
            procedure=procedure,
            outcome=outcome,
            reusable_rule=reusable_rule,
            failure_modes=failure_modes,
        )

    @staticmethod
    def _load_json_object(response: str) -> Dict[str, Any]:
        text = response.strip()
        if not text:
            raise ValueError("Empty tactical summary response")

        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"Unable to parse tactical summary response: {response!r}")
            loaded = json.loads(text[start : end + 1])

        if not isinstance(loaded, dict):
            raise ValueError(f"Tactical summary response must be a JSON object: {response!r}")
        return loaded

    @staticmethod
    def _coerce_optional_str(value: object) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _coerce_str_list(value: object) -> List[str]:
        if not isinstance(value, list):
            return []
        items: List[str] = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    items.append(stripped)
        return items


__all__ = [
    "TACTICAL_FORMATION_PROMPT",
    "TACTICAL_SUMMARY_PROMPT",
    "TacticalFormationCandidate",
    "TacticalFormationDecision",
    "TacticalFormationJudge",
    "TacticalSummaryDraft",
    "TacticalSummaryWriter",
]
