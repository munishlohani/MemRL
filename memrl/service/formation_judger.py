"""Tactical formation candidate model and LLM-backed procedural summarizer.

There is no LLM judgment step here anymore: the phase-1 advantage gate
(EpisodeRunner._should_queue_tactical_candidate) is the sole formation
decision. The LLM's only remaining role is TacticalSummaryWriter, which
turns an admitted candidate into a reusable procedural memory -- it never
decides whether to keep the candidate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..providers.base import BaseLLM
import logging


logger = logging.getLogger(__name__)


@dataclass
class TacticalFormationCandidate:
    """One above-baseline experience (Stage-1 advantage gate, §4.1) admitted for storage; awaiting summarization."""

    candidate_id: str
    task_type: str
    task_description: str
    episode_id: str
    episode_index: int
    step_index: int
    observation: str
    action: str
    reward: float
    advantage: float
    history: str
    retrieved_memories: str
    episode_slot_index: Optional[int] = None
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
            f"episode_slot_index: {self.episode_slot_index if self.episode_slot_index is not None else 'none'}",
            f"step_index: {self.step_index}",
            f"observation: {self.observation}",
            f"action: {self.action}",
            f"advantage (this step's return-to-go minus the task-type baseline -- why it was admitted): {self.advantage:.6f}",
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
            f" (advantage={self.advantage:.3f})"
        ).strip()


@dataclass
class TacticalSummaryDraft:
    """Reusable tactical procedure: goal + concrete ordered steps + outcome.

    Unlike the strategic layer (which abstracts over the object, §17.4),
    tactical content is deliberately object-SPECIFIC -- it names the actual
    object/receptacle/appliance involved, since that's what makes a d=2
    node a distinct, retrievable skill rather than a duplicate of every
    other instance of the same abstract task. It is still a distilled
    procedure, not a verbatim step-by-step transcript: a live run showed
    that form doing more harm than good by over-fitting retrieval to the
    one instance the memory happened to be formed from.
    """

    goal: str
    procedure: List[str] = field(default_factory=list)
    outcome: str = ""

    def to_text(self) -> str:
        """Render the draft into a canonical procedural memory string."""
        lines: List[str] = []
        if self.goal.strip():
            lines.append(f"GOAL: {self.goal.strip()}")
        if self.procedure:
            lines.append("PROCEDURE:")
            for idx, step in enumerate(self.procedure, 1):
                step_text = step.strip()
                if step_text:
                    lines.append(f"{idx}. {step_text}")
        if self.outcome.strip():
            lines.append(f"OUTCOME: {self.outcome.strip()}")
        return "\n".join(lines).strip()


TACTICAL_SUMMARY_PROMPT = """You are converting one tactical step into a reusable, grounded tactical note.

The note will be embedded and retrieved later when the agent faces a similar situation. Focus on THIS step -- the action taken and the observation it responded to. The prior history is given only as context; do not summarize the whole episode. Keep the actual objects, receptacles, and appliances named explicitly (e.g. "the tomato", "the fridge") -- do NOT replace them with generic placeholders like "the target object" or "the receptacle". This is the tactical layer: it should read as a specific, grounded skill, not an abstract strategy. Describe only what actually happened; do not fabricate a different action or observation than the one given below.

The `procedure` steps must be the EXACT, LITERAL environment commands as they were issued -- copy the `action` string (and any immediately preceding actions from history that set it up) verbatim, character for character, including object/receptacle ids and any fixed syntax like "in/on". These notes are replayed directly as commands, so a paraphrase (e.g. "place the knife on the counter" instead of "put butterknife 1 in/on countertop 1") is useless -- it will not match a valid command and the step will fail. Do not rephrase, normalize, or "clean up" the command text.

Return a single JSON object and nothing else.

Schema:
{{
  "goal": string,
  "procedure": [string, ...],
  "outcome": string
}}

Every step here comes from a positive experience -- an action that worked. So write ONLY affirmative commands the agent should DO ("use the microwave", "put butterknife 1 in/on countertop 1"). Never write a negation, prohibition, or warning ("cannot use the microwave", "do not open the fridge", "avoid the sink") -- those are not valid commands, cannot be replayed, and have no place in a positive skill.

Rules:
- goal: one sentence describing the situation this step handles, naming the actual object/receptacle/appliance involved.
- procedure: an ordered list of the EXACT literal commands that were executed (copied verbatim from `action` and the setup actions in history), in the order they were issued. Do not paraphrase or generalize -- these are replayed as-is. Affirmative commands only; never a negation or prohibition.
- outcome: one short sentence on what this step accomplished in the environment. Do not state "success"/"failure" -- a single step is not the episode's outcome.
- Do not include markdown, explanations, or extra keys.

Source experience:
{candidate}
"""


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

        goal = self._coerce_optional_str(payload.get("goal")) or candidate.task_description
        procedure = self._coerce_procedure(payload.get("procedure"))
        if not procedure:
            procedure = [f"{candidate.action.strip()} (given: {candidate.observation.strip()})"]
        outcome = self._coerce_optional_str(payload.get("outcome")) or f"Reward={candidate.reward:.3f}"

        return TacticalSummaryDraft(
            goal=goal,
            procedure=procedure,
            outcome=outcome,
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
    def _coerce_procedure(value: object) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


__all__ = [
    "TACTICAL_SUMMARY_PROMPT",
    "TacticalFormationCandidate",
    "TacticalSummaryDraft",
    "TacticalSummaryWriter",
]
