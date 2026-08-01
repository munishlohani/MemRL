"""LifelongBench (LLB) adapter for the benchmark-neutral EpisodeRunner.

LLB (vendored under 3rdparty/LifelongAgentBench) covers two task types --
DBBench (SQL against a live MySQL Docker container) and OSInteraction
(bash against a live Docker sandbox) -- each driven by a Session/Task state
machine, not a simple step-once environment. This adapter builds a fresh
Task+Session per active batch slot at reset() (matching the original
llb_rl_runner.py's per-sample "fresh task object" pattern), injects the
agent's raw response as the Session's next AGENT chat-history item at
step(), and lets the vendored Task.interact()/complete() do the real work
(SQL/bash execution, grading) -- this adapter never re-implements that
parsing/grading logic itself (see memrl/agent/llb_agent.py's module
docstring for why).

Docker is required for BOTH task types to actually run reset()/step():
DBBench's Docker container is constructed inside build_task() itself (its
Task.__init__ builds the container immediately), so build_task("db_bench",
...) fails outright without a running Docker daemon; OSInteraction defers
container creation to its own _reset(), so the Task object can be built
without Docker but reset()/step() will still fail. Dataset loading/listing
(num_tasks, known_task_types, train/val/test pool bookkeeping) never
touches Docker -- it reads the raw JSON files directly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from memrl.lifelongbench_eval.task_wrappers import (
    ChatHistoryItem,
    Role,
    SampleStatus,
    SessionEvaluationOutcome,
    Session,
    build_task,
    sorted_sample_indices,
)

from ..episode.env_adapter import EpisodeEnvAdapter, EpisodeResetResult, EpisodeStepResult

logger = logging.getLogger(__name__)


def _shorten_text(text: Any, *, limit: int = 160) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _latest_observation(session: Session) -> str:
    try:
        if session.chat_history.get_value_length() == 0:
            return ""
        return str(session.chat_history.get_item_deep_copy(-1).content or "")
    except Exception:
        return ""


class LLBEpisodeEnvAdapter(EpisodeEnvAdapter):
    """Adapt LifelongBench (one task type per instance) to the EpisodeRunner interface.

    A single adapter instance serves exactly one task type ("db_bench" or
    "os_interaction") -- DB and OS need different Docker images/containers
    entirely, and the original LLBRunner never mixed them in one run either
    (one `--task` CLI arg per invocation, same convention as BCB's
    `--subset`).
    """

    _PHASES = ("train", "val", "test")

    def __init__(
        self,
        *,
        task: str,
        train_file: str,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        batch_size: int = 1,
        max_steps: int = 15,
        os_timeout: int = 20,
    ) -> None:
        self._task = str(task)
        self._data_file_by_phase: Dict[str, Optional[str]] = {
            "train": train_file,
            "val": val_file,
            "test": test_file,
        }
        self._batch_size = max(1, int(batch_size))
        self._max_steps = max(1, int(max_steps))
        self._os_timeout = int(os_timeout)

        self._phase = "train"
        self._loaded_phases: Dict[str, bool] = {"train": False, "val": False, "test": False}
        self._pools: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        self._cursors: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        # Mirrors AlfWorldEpisodeEnvAdapter/BCBEpisodeEnvAdapter's exact-count
        # epoch coverage pattern: a wrap-around slot (pool size not evenly
        # divisible by batch_size) is marked duplicate so EpisodeRunner
        # excludes it from aggregated metrics instead of double-counting.
        self._seen_ids: Dict[str, set] = {"train": set(), "val": set(), "test": set()}

        self._slots: Dict[int, Dict[str, Any]] = {}
        self._last_reset_sample_ids: List[Optional[str]] = []

    def _pool_for(self, phase: str) -> List[str]:
        if phase not in self._PHASES:
            raise ValueError(f"Unknown LLB phase: {phase!r} (expected one of {self._PHASES})")
        return self._pools[phase]

    def _ensure_loaded(self, phase: str) -> None:
        if self._loaded_phases.get(phase):
            return
        data_file = self._data_file_by_phase.get(phase)
        if data_file:
            self._pools[phase] = sorted_sample_indices(data_file, limit=None)
        else:
            self._pools[phase] = []
        self._loaded_phases[phase] = True

    def set_phase(self, phase: str) -> None:
        """Switch the active task pool among 'train', 'val', and 'test' --
        mirrors BCBEpisodeEnvAdapter.set_phase(). run_llb.py flips this
        before/after a validation or frozen-test pass."""
        if phase not in self._PHASES:
            raise ValueError(f"Unknown LLB phase: {phase!r} (expected one of {self._PHASES})")
        self._phase = phase

    def reset_epoch_tracking(self) -> None:
        """Clear the seen-ids set for the CURRENT phase. Call once per
        epoch/pass (same convention as BCBEpisodeEnvAdapter)."""
        if self._phase in self._seen_ids:
            self._seen_ids[self._phase] = set()

    def num_tasks(self) -> int:
        self._ensure_loaded("train")
        return len(self._pools["train"])

    def num_val_tasks(self) -> int:
        self._ensure_loaded("val")
        return len(self._pools["val"])

    def num_test_tasks(self) -> int:
        self._ensure_loaded("test")
        return len(self._pools["test"])

    def known_task_types(self) -> List[str]:
        # LLB has no per-sample taxonomy the way ALFWorld/BCB do -- every
        # sample in one adapter instance is the same task type.
        return [self._task]

    def is_batch(self) -> bool:
        return self._batch_size > 1

    def reset(self, **kwargs: Any) -> EpisodeResetResult:
        self._ensure_loaded(self._phase)
        pool = self._pool_for(self._phase)
        observations: List[str] = []
        infos: List[Dict[str, Any]] = []
        sample_ids: List[Optional[str]] = []
        self._slots = {}

        if not pool:
            self._last_reset_sample_ids = []
            return EpisodeResetResult(observations=[], infos=[])

        data_file = self._data_file_by_phase.get(self._phase)
        for slot_idx in range(self._batch_size):
            sample_id = pool[self._cursors[self._phase] % len(pool)]
            self._cursors[self._phase] += 1
            seen = self._seen_ids.get(self._phase)
            is_duplicate = seen is not None and sample_id in seen
            if seen is not None and not is_duplicate:
                seen.add(sample_id)

            try:
                task_obj, task_name = build_task(
                    task=self._task,
                    data_file_path=data_file,
                    max_round=self._max_steps,
                    os_timeout=self._os_timeout,
                )
                session = Session(task_name=task_name, sample_index=sample_id)
                task_obj.reset(session)
            except Exception as exc:
                logger.error("LLB reset failed for sample_id=%s: %s", sample_id, exc, exc_info=True)
                sample_ids.append(sample_id)
                observations.append("")
                infos.append(
                    {
                        "episode_id": sample_id,
                        "task_id": sample_id,
                        "task_type": self._task,
                        "phase": self._phase,
                        "duplicate": is_duplicate,
                        "reset_error": str(exc),
                    }
                )
                continue

            observation = _latest_observation(session)
            self._slots[slot_idx] = {
                "task_obj": task_obj,
                "session": session,
                "sample_id": sample_id,
            }
            sample_ids.append(sample_id)
            observations.append(observation)
            infos.append(
                {
                    "episode_id": sample_id,
                    "task_id": sample_id,
                    "task_description": observation,
                    "task_type": self._task,
                    "phase": self._phase,
                    "duplicate": is_duplicate,
                }
            )

        self._last_reset_sample_ids = sample_ids
        self._log_reset_output(observations, infos)
        return EpisodeResetResult(observations=observations, infos=infos)

    def step(self, actions: List[Any], **kwargs: Any) -> EpisodeStepResult:
        observations: List[str] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []

        for slot_idx, action in enumerate(actions):
            slot = self._slots.get(slot_idx)
            if slot is None:
                observations.append("")
                rewards.append(0.0)
                dones.append(True)
                infos.append({"status": "NO_TASK", "error": "no task for this slot"})
                continue

            task_obj = slot["task_obj"]
            session: Session = slot["session"]
            sample_id = slot["sample_id"]
            raw_response = str(action or "")

            try:
                session.chat_history.inject(
                    ChatHistoryItem(role=Role.AGENT, content=raw_response)
                )
                task_obj.interact(session)
            except Exception as exc:
                # Task.interact's own exception handling for unexpected
                # (non-TaskEnvironmentException) errors re-raises rather
                # than absorbing into a SampleStatus (see module docstring
                # in llb_agent.py / plan notes) -- treat that as a failed,
                # done episode rather than crashing the whole batch step.
                logger.error(
                    "LLB interact failed for sample_id=%s: %s", sample_id, exc, exc_info=True
                )
                observations.append(f"LLB interact error: {exc}")
                rewards.append(0.0)
                dones.append(True)
                infos.append(
                    {
                        "episode_id": sample_id,
                        "task_type": self._task,
                        "phase": self._phase,
                        "status": "INTERACT_ERROR",
                        "error": str(exc),
                    }
                )
                self._release_slot(slot_idx)
                continue

            done = session.sample_status != SampleStatus.RUNNING
            reward = 0.0
            if done:
                try:
                    task_obj.complete(session)
                    reward = (
                        1.0
                        if session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
                        else 0.0
                    )
                except Exception as exc:
                    logger.error(
                        "LLB complete failed for sample_id=%s: %s", sample_id, exc, exc_info=True
                    )

            observation = _latest_observation(session)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(
                {
                    "episode_id": sample_id,
                    "task_type": self._task,
                    "phase": self._phase,
                    "sample_status": str(session.sample_status),
                    "finish_reason": session.finish_reason,
                }
            )

            if done:
                self._release_slot(slot_idx)

        self._log_step_output(actions, observations, rewards, dones, infos)
        return EpisodeStepResult(observations=observations, rewards=rewards, dones=dones, infos=infos)

    def _release_slot(self, slot_idx: int) -> None:
        slot = self._slots.pop(slot_idx, None)
        if slot is None:
            return
        task_obj = slot.get("task_obj")
        release = getattr(task_obj, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                logger.debug("LLB task release failed for slot %s", slot_idx, exc_info=True)

    def close(self) -> None:
        for slot_idx in list(self._slots.keys()):
            self._release_slot(slot_idx)

    def episode_id(self, index: int = 0) -> Optional[str]:
        if index >= len(self._last_reset_sample_ids):
            return None
        return self._last_reset_sample_ids[index]

    def task_type(self, index: int = 0) -> Optional[str]:
        return self._task

    def _log_reset_output(self, observations: List[str], infos: List[Dict[str, Any]]) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        for idx, observation in enumerate(observations):
            info = infos[idx] if idx < len(infos) else {}
            logger.info(
                "LLB reset[%s]: task_type=%s sample_id=%s obs=%s",
                idx,
                self._task,
                info.get("episode_id"),
                _shorten_text(observation),
            )

    def _log_step_output(
        self,
        actions: List[Any],
        observations: List[str],
        rewards: List[float],
        dones: List[bool],
        infos: List[Dict[str, Any]],
    ) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        for idx, action in enumerate(actions):
            reward = rewards[idx] if idx < len(rewards) else 0.0
            done = dones[idx] if idx < len(dones) else False
            info = infos[idx] if idx < len(infos) else {}
            logger.info(
                "LLB step[%s]: sample_id=%s status=%s reward=%.3f done=%s",
                idx,
                info.get("episode_id"),
                info.get("sample_status") or info.get("status"),
                float(reward),
                bool(done),
            )


__all__ = [
    "LLBEpisodeEnvAdapter",
    "EpisodeResetResult",
    "EpisodeStepResult",
]
