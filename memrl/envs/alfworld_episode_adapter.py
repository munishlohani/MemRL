"""ALFWorld adapter for the benchmark-neutral EpisodeRunner.

Wraps :class:`memrl.envs.alfworld_env.AlfWorldEnv` (textual ALFWorld) and
translates its ``list[dict]`` reset/step outputs into the normalized
``EpisodeResetResult`` / ``EpisodeStepResult`` shapes the EpisodeRunner
consumes. This is the per-benchmark EpisodeEnvAdapter the public runner
lacked (Reviewer W2): with it, ``run/run_alfworld.py`` can drive the new
agentic two-tier EpisodeRunner instead of the legacy flat-RAG AlfworldRunner.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from ..episode.env_adapter import EpisodeEnvAdapter, EpisodeResetResult, EpisodeStepResult

logger = logging.getLogger(__name__)


_ALFWORLD_TASK_PREFIXES = (
    "pick_and_place",
    "pick_clean_then_place",
    "pick_heat_then_place",
    "pick_cool_then_place",
    "look_at_obj",
    "pick_two",
)


def _task_type_from_gamefile(gamefile: Optional[str]) -> Optional[str]:
    """Derive an ALFWorld task type from a gamefile path.

    ALFWorld gamefiles live in directories whose name starts with one of the
    six canonical task prefixes (e.g. ``pick_and_place_simple-...``). We map
    any matching prefix to the canonical task type; this is the per-task-type
    signal the Phase-1 Q-value machinery keys on (spec §3.1).
    """
    if not gamefile:
        return None
    directory = os.path.basename(os.path.dirname(os.path.normpath(gamefile)))
    for prefix in _ALFWORLD_TASK_PREFIXES:
        if directory.startswith(prefix):
            return prefix
    return None


def _task_description_from_observation(observation: Optional[str]) -> Optional[str]:
    text = str(observation or "")
    patterns = (
        r"Your task is to:\s*(.+)",
        r"Task:\s*(.+)",
        r"Your goal is to:\s*(.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            task = match.group(1).strip()
            if task:
                task = task.split("\n", 1)[0].strip()
                return task.rstrip(".")
    return None


def _shorten_text(text: Any, *, limit: int = 160) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


class AlfWorldEpisodeEnvAdapter(EpisodeEnvAdapter):
    """Adapt a (possibly batched) AlfWorldEnv to the EpisodeRunner interface.

    The underlying AlfWorldEnv is constructed lazily so that the runner can be
    imported and unit-tested on machines where ALFWorld is not installed; the
    import only fails when ``reset`` is actually called without a preconfigured
    environment.
    """

    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        alf_env: Optional[Any] = None,
        task_type: str = "train",
        batch_size: int = 1,
    ) -> None:
        self._config_path = config_path
        self._task_type = task_type
        self._batch_size = max(1, int(batch_size))
        self._alf_env = alf_env
        self._last_reset_infos: List[Dict[str, Any]] = []

    @property
    def alf_env(self) -> Any:
        if self._alf_env is None:
            from .alfworld_env import AlfWorldEnv  # lazy: only need ALFWorld at runtime

            assert self._config_path, "config_path is required when no alf_env is supplied"
            self._alf_env = AlfWorldEnv(
                config_path=self._config_path,
                task_type=self._task_type,
                batch_size=self._batch_size,
            )
        return self._alf_env

    def reset(self, **kwargs: Any) -> EpisodeResetResult:
        raw = self.alf_env.reset()
        observations: List[str] = []
        infos: List[Dict[str, Any]] = []
        for entry in raw:
            observations.append(str(entry.get("obs", "")))
            info = dict(entry.get("info") or {})
            gamefile = info.get("gamefile")
            task_type = _task_type_from_gamefile(gamefile)
            if task_type is not None:
                info.setdefault("task_type", task_type)
            task_description = _task_description_from_observation(info.get("obs") or entry.get("obs"))
            if task_description is not None:
                info.setdefault("task_description", task_description)
            infos.append(info)
        self._last_reset_infos = infos
        self._log_reset_output(observations, infos)
        return EpisodeResetResult(observations=observations, infos=infos)

    def step(self, actions: List[Any], **kwargs: Any) -> EpisodeStepResult:
        raw = self.alf_env.step(list(actions))
        observations: List[str] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []
        for idx, entry in enumerate(raw):
            observations.append(str(entry.get("obs", "")))
            rewards.append(float(entry.get("reward", 0.0) or 0.0))
            dones.append(bool(entry.get("done", False)))
            info = dict(entry.get("info") or {})
            reset_info = (
                self._last_reset_infos[idx]
                if idx < len(self._last_reset_infos)
                else {}
            )
            # Carry forward reset-time identity so the runner can resolve
            # episode_id / task_type on every step, not just at reset.
            for key in ("gamefile", "task_type"):
                if key not in info and key in reset_info:
                    info[key] = reset_info[key]
            if "task_description" not in info:
                task_description = _task_description_from_observation(observations[-1] if observations else entry.get("obs"))
                if task_description is not None:
                    info["task_description"] = task_description
            infos.append(info)
        self._log_step_output(actions, observations, rewards, dones, infos)
        return EpisodeStepResult(
            observations=observations,
            rewards=rewards,
            dones=dones,
            infos=infos,
        )

    def close(self) -> None:
        alf_env = self._alf_env
        if alf_env is None:
            return
        try:
            alf_env.close()
        except Exception:
            logger.debug("AlfWorldEnv.close failed", exc_info=True)
        finally:
            self._alf_env = None
            self._last_reset_infos = []

    def episode_id(self, index: int = 0) -> Optional[str]:
        if index >= len(self._last_reset_infos):
            return None
        gamefile = self._last_reset_infos[index].get("gamefile")
        if isinstance(gamefile, str) and gamefile:
            return os.path.normpath(gamefile)
        return None

    def task_type(self, index: int = 0) -> Optional[str]:
        if index >= len(self._last_reset_infos):
            return None
        return self._last_reset_infos[index].get("task_type")

    def is_batch(self) -> bool:
        return self._batch_size > 1

    def _log_reset_output(
        self,
        observations: List[str],
        infos: List[Dict[str, Any]],
    ) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        for idx, observation in enumerate(observations):
            info = infos[idx] if idx < len(infos) else {}
            logger.info(
                "ALFWorld reset[%s]: task_type=%s gamefile=%s obs=%s",
                idx,
                info.get("task_type"),
                info.get("gamefile"),
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
            observation = observations[idx] if idx < len(observations) else ""
            reward = rewards[idx] if idx < len(rewards) else 0.0
            done = dones[idx] if idx < len(dones) else False
            info = infos[idx] if idx < len(infos) else {}
            logger.info(
                "ALFWorld step[%s]: action=%s reward=%.3f done=%s task_type=%s obs=%s",
                idx,
                _shorten_text(action),
                float(reward),
                bool(done),
                info.get("task_type"),
                _shorten_text(observation),
            )


__all__ = [
    "AlfWorldEpisodeEnvAdapter",
    "EpisodeResetResult",
    "EpisodeStepResult",
    "_task_description_from_observation",
    "_task_type_from_gamefile",
]
