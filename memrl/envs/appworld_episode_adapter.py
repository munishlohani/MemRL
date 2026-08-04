"""AppWorld adapter for the benchmark-neutral EpisodeRunner.

AppWorld runs behind a subprocess boundary (see memrl/appworld_eval/worker.py):
it pins pydantic 1.x / SQLAlchemy 1.4 and cannot share MemRL's interpreter, so
one AppWorldClient per batch slot drives one worker process, each holding one
live task at a time. That mirrors LLBEpisodeEnvAdapter, which likewise owns an
external resource (a Docker container) per slot.

Reward is task-goal completion as a binary: 1.0 when every requirement in
AppWorld's TestTracker passes, else 0.0. That matches AppWorld's headline TGC
metric and the 0.0/1.0 convention the other three adapters already use.
pass_percentage / difficulty / num_tests ride along in `info` for analysis but
never enter the reward.

Two AppWorld specifics worth knowing:

- Completion is agent-signalled. The episode is not finished until the agent
  calls `apis.supervisor.complete_task(...)`, which the worker reports as
  `task_completed`. There is no environment-side "you win" signal mid-episode.
- AppWorld's own `max_interactions` defaults to 1000, which is not a useful
  cap. The real ceiling is `max_steps` from the experiment config, enforced
  here, so a stuck episode ends in tens of turns rather than hundreds.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from memrl.appworld_eval.client import AppWorldClient, AppWorldClientError

from ..episode.env_adapter import EpisodeEnvAdapter, EpisodeResetResult, EpisodeStepResult

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(action: Any) -> str:
    """Pull runnable Python out of the agent's action text.

    AppWorldAgent forwards the whole `Action:` payload, which normally is a
    fenced block. Strip the fence when present; otherwise send the text as-is
    (a model that emitted bare code should still get to run it).
    """
    text = str(action or "").strip()
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    if match:
        return (match.group(1) or "").strip()
    return text


def _shorten(text: Any, *, limit: int = 160) -> str:
    value = str(text or "").strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


class AppWorldEpisodeEnvAdapter(EpisodeEnvAdapter):
    """Adapt AppWorld to the EpisodeRunner interface."""

    _PHASES = ("train", "val", "test")

    def __init__(
        self,
        *,
        appworld_root: str,
        appworld_python: str,
        train_split: str = "train",
        val_split: str = "dev",
        test_split: str = "test_normal",
        batch_size: int = 1,
        max_steps: int = 30,
        experiment_name: str = "memrl",
    ) -> None:
        self._root = str(appworld_root)
        self._python = str(appworld_python)
        self._split_by_phase: Dict[str, str] = {
            "train": str(train_split),
            "val": str(val_split),
            "test": str(test_split),
        }
        self._batch_size = max(1, int(batch_size))
        self._max_steps = max(1, int(max_steps))
        self._experiment_name = str(experiment_name)

        self._phase = "train"
        self._pools: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        self._loaded: Dict[str, bool] = {"train": False, "val": False, "test": False}
        self._cursors: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        # Same exact-count epoch coverage as the other adapters: a wrap-around
        # slot is flagged duplicate so EpisodeRunner leaves it out of metrics.
        self._seen: Dict[str, set] = {"train": set(), "val": set(), "test": set()}

        self._clients: Dict[int, AppWorldClient] = {}
        self._slots: Dict[int, Dict[str, Any]] = {}
        self._last_reset_task_ids: List[Optional[str]] = []

    # -- worker pool ---------------------------------------------------------

    def _client(self, slot_idx: int) -> AppWorldClient:
        """One long-lived worker per slot, reused across episodes -- importing
        appworld and booting its app code is slow, so paying it per episode
        would dominate runtime."""
        client = self._clients.get(slot_idx)
        if client is None:
            client = AppWorldClient(
                python_executable=self._python,
                appworld_root=self._root,
                experiment_name=self._experiment_name,
            )
            self._clients[slot_idx] = client
        return client

    def _ensure_loaded(self, phase: str) -> None:
        if self._loaded.get(phase):
            return
        split = self._split_by_phase.get(phase) or ""
        if not split:
            self._pools[phase] = []
        else:
            # Slot 0's worker doubles as the task-list source; listing task ids
            # touches no app state.
            self._pools[phase] = self._client(0).task_ids(split)
        self._loaded[phase] = True

    # -- EpisodeEnvAdapter surface -------------------------------------------

    def set_phase(self, phase: str) -> None:
        if phase not in self._PHASES:
            raise ValueError(f"Unknown AppWorld phase: {phase!r} (expected {self._PHASES})")
        self._phase = phase

    def reset_epoch_tracking(self) -> None:
        if self._phase in self._seen:
            self._seen[self._phase] = set()

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
        # AppWorld tasks share one task type here. Its own scenario/difficulty
        # taxonomy is carried per-episode in `info` instead of being used to
        # partition the memory graph.
        return ["appworld"]

    def is_batch(self) -> bool:
        return self._batch_size > 1

    def reset(self, **kwargs: Any) -> EpisodeResetResult:
        self._ensure_loaded(self._phase)
        pool = self._pools[self._phase]
        observations: List[str] = []
        infos: List[Dict[str, Any]] = []
        task_ids: List[Optional[str]] = []
        self._slots = {}

        if not pool:
            self._last_reset_task_ids = []
            return EpisodeResetResult(observations=[], infos=[])

        for slot_idx in range(self._batch_size):
            task_id = pool[self._cursors[self._phase] % len(pool)]
            self._cursors[self._phase] += 1
            seen = self._seen.get(self._phase)
            is_duplicate = seen is not None and task_id in seen
            if seen is not None and not is_duplicate:
                seen.add(task_id)

            try:
                info = self._client(slot_idx).reset(task_id)
            except AppWorldClientError as exc:
                logger.error("AppWorld reset failed for task_id=%s: %s", task_id, exc)
                task_ids.append(task_id)
                observations.append("")
                infos.append(
                    {
                        "episode_id": task_id,
                        "task_id": task_id,
                        "task_type": "appworld",
                        "phase": self._phase,
                        "duplicate": is_duplicate,
                        "reset_error": str(exc),
                    }
                )
                continue

            instruction = str(info.get("instruction") or "")
            self._slots[slot_idx] = {"task_id": task_id, "steps": 0}
            task_ids.append(task_id)
            # The agent's own task_description carries the instruction; the
            # observation slot is the previous code output, empty at reset.
            observations.append("")
            infos.append(
                {
                    "episode_id": task_id,
                    "task_id": task_id,
                    "task_description": instruction,
                    "task_type": "appworld",
                    "phase": self._phase,
                    "duplicate": is_duplicate,
                }
            )

        self._last_reset_task_ids = task_ids
        if logger.isEnabledFor(logging.INFO):
            for idx, info in enumerate(infos):
                logger.info(
                    "AppWorld reset[%s]: task_id=%s instruction=%s",
                    idx, info.get("task_id"), _shorten(info.get("task_description")),
                )
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

            task_id = slot["task_id"]
            code = _extract_code(action)

            try:
                result = self._client(slot_idx).execute(code)
            except AppWorldClientError as exc:
                logger.error("AppWorld execute failed for task_id=%s: %s", task_id, exc)
                observations.append(f"AppWorld execution error: {exc}")
                rewards.append(0.0)
                dones.append(True)
                infos.append(
                    {
                        "episode_id": task_id,
                        "task_type": "appworld",
                        "phase": self._phase,
                        "status": "EXECUTE_ERROR",
                        "error": str(exc),
                    }
                )
                self._release_slot(slot_idx)
                continue

            slot["steps"] += 1
            output = str(result.get("output") or "")
            completed = bool(result.get("task_completed"))
            # AppWorld's own max_interactions is 1000; the effective cap is ours.
            budget_exhausted = slot["steps"] >= self._max_steps
            done = completed or budget_exhausted

            reward = 0.0
            info: Dict[str, Any] = {
                "episode_id": task_id,
                "task_type": "appworld",
                "phase": self._phase,
                "steps": slot["steps"],
                "task_completed": completed,
                "num_interactions": result.get("num_interactions"),
            }

            if done:
                # Scored once, at the end: TestTracker only reports a verdict
                # for the world as it now stands.
                try:
                    evaluation = self._client(slot_idx).evaluate()
                    reward = 1.0 if evaluation.get("success") else 0.0
                    info.update(
                        {
                            "success": bool(evaluation.get("success")),
                            "pass_count": evaluation.get("pass_count"),
                            "fail_count": evaluation.get("fail_count"),
                            "num_tests": evaluation.get("num_tests"),
                            "pass_percentage": evaluation.get("pass_percentage"),
                            "difficulty": evaluation.get("difficulty"),
                        }
                    )
                except AppWorldClientError as exc:
                    logger.error("AppWorld evaluate failed for task_id=%s: %s", task_id, exc)
                    info["evaluate_error"] = str(exc)
                if budget_exhausted and not completed:
                    info["status"] = "STEP_BUDGET_EXHAUSTED"

            observations.append(output)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            if done:
                self._release_slot(slot_idx)

        if logger.isEnabledFor(logging.INFO):
            for idx, info in enumerate(infos):
                logger.info(
                    "AppWorld step[%s]: task_id=%s steps=%s completed=%s reward=%.3f done=%s",
                    idx,
                    info.get("episode_id"),
                    info.get("steps"),
                    info.get("task_completed"),
                    float(rewards[idx]) if idx < len(rewards) else 0.0,
                    bool(dones[idx]) if idx < len(dones) else False,
                )
        return EpisodeStepResult(
            observations=observations, rewards=rewards, dones=dones, infos=infos
        )

    def _release_slot(self, slot_idx: int) -> None:
        """Close the finished task but KEEP the worker process -- the next
        episode in this slot reuses it."""
        self._slots.pop(slot_idx, None)
        client = self._clients.get(slot_idx)
        if client is None:
            return
        try:
            client.close_world()
        except Exception:
            logger.debug("AppWorld close_world failed for slot %s", slot_idx, exc_info=True)

    def close(self) -> None:
        for slot_idx in list(self._slots.keys()):
            self._release_slot(slot_idx)
        for slot_idx, client in list(self._clients.items()):
            try:
                client.close()
            except Exception:
                logger.debug("AppWorld client close failed for slot %s", slot_idx, exc_info=True)
        self._clients = {}

    def episode_id(self, index: int = 0) -> Optional[str]:
        if index >= len(self._last_reset_task_ids):
            return None
        return self._last_reset_task_ids[index]

    def task_type(self, index: int = 0) -> Optional[str]:
        return "appworld"


__all__ = [
    "AppWorldEpisodeEnvAdapter",
    "EpisodeResetResult",
    "EpisodeStepResult",
]
