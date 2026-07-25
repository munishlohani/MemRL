"""BigCodeBench (BCB) adapter for the benchmark-neutral EpisodeRunner.

Each BCB coding task is framed as a 1-step episode: reset() presents the
task prompt as the observation, step() evaluates the agent's submitted code
against the official BigCodeBench untrusted_check harness and returns
done=True unconditionally (reward=1.0 on PASS, else 0.0). This lets BCB
reuse the full two-tier skill graph (retrieval, tactical/strategic
Q-learning, sleep consolidation) with zero changes to EpisodeRunner itself.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from memrl.bigcodebench_eval.bcb_adapter import extract_code_from_response
from memrl.bigcodebench_eval.eval_utils import (
    ensure_bigcodebench_on_path,
    run_untrusted_check_with_hard_timeout,
    sanitize_code,
)
from memrl.bigcodebench_eval.task_wrappers import get_prompt, load_bcb_data, split_dataset

from ..episode.env_adapter import EpisodeEnvAdapter, EpisodeResetResult, EpisodeStepResult

logger = logging.getLogger(__name__)


def _shorten_text(text: Any, *, limit: int = 160) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


class BCBEpisodeEnvAdapter(EpisodeEnvAdapter):
    """Adapt BigCodeBench task selection + evaluation to the EpisodeRunner interface.

    Data is loaded lazily (on first reset()/known_task_types() call) so the
    adapter can be constructed and unit-tested without the BigCodeBench
    dataset or vendored eval harness present.
    """

    _PHASES = ("train", "val", "test")

    def __init__(
        self,
        *,
        subset: str = "full",
        split: str = "instruct",
        data_path: Optional[str] = None,
        split_file: Optional[str] = None,
        train_ratio: float = 0.7,
        test_ratio: float = 0.0,
        seed: int = 42,
        batch_size: int = 1,
        bcb_repo: Optional[str] = None,
        eval_timeout_s: float = 60.0,
        untrusted_hard_timeout_s: float = 120.0,
    ) -> None:
        self._subset = subset
        self._split = split
        self._data_path = data_path
        self._split_file = split_file
        self._train_ratio = float(train_ratio)
        # 0.0 (default) reproduces the original two-way train/val split
        # exactly -- a frozen test pool only exists if the caller opts in,
        # either via this ratio or a split_file whose JSON defines
        # non-empty test_ids (see split_dataset).
        self._test_ratio = float(test_ratio)
        self._seed = int(seed)
        self._batch_size = max(1, int(batch_size))
        self._bcb_repo = bcb_repo
        self._eval_timeout_s = float(eval_timeout_s)
        self._untrusted_hard_timeout_s = float(untrusted_hard_timeout_s)

        self._loaded = False
        self._problems: Dict[str, Dict[str, Any]] = {}
        self._train_ids: List[str] = []
        self._val_ids: List[str] = []
        self._test_ids: List[str] = []
        self._task_type_by_id: Dict[str, str] = {}
        self._phase = "train"
        self._cursors: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        # Tracks task_ids already dispatched this pass, per non-train
        # phase, so a wrap-around slot (pool size not evenly divisible by
        # batch_size) can be marked duplicate and excluded from aggregated
        # eval metrics -- mirrors AlfWorldEpisodeEnvAdapter's
        # _seen_gamefiles/reset_epoch_tracking pattern for exact-count
        # coverage. Train phase never marks duplicates (it cycles forever).
        self._seen_ids: Dict[str, set] = {"val": set(), "test": set()}
        self._last_reset_task_ids: List[Optional[str]] = []

    def _pool_for(self, phase: str) -> List[str]:
        if phase == "train":
            return self._train_ids
        if phase == "val":
            return self._val_ids
        if phase == "test":
            return self._test_ids
        raise ValueError(f"Unknown BCB phase: {phase!r} (expected one of {self._PHASES})")

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        ensure_bigcodebench_on_path(self._bcb_repo)
        self._problems = load_bcb_data(subset=self._subset, data_path=self._data_path)
        self._train_ids, self._val_ids, self._test_ids = split_dataset(
            self._problems,
            train_ratio=self._train_ratio,
            test_ratio=self._test_ratio,
            seed=self._seed,
            split_file=self._split_file,
        )
        self._task_type_by_id = self._derive_task_types(self._problems)
        self._loaded = True

    @staticmethod
    def _derive_task_types(problems: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Each task's type = its least-frequent (most distinctive) library
        across the loaded subset, or "stdlib" if it uses none. BCB has no
        fixed taxonomy like ALFWorld's six task types -- `libs` is the only
        quasi-categorical signal available, and picking the rarest library
        avoids everything collapsing onto a handful of common ones (e.g.
        `os`, `re`) while avoiding the combinatorial explosion of keying on
        the full library set.
        """
        freq: Counter[str] = Counter()
        for task in problems.values():
            for lib in task.get("libs") or []:
                freq[str(lib)] += 1

        task_type_by_id: Dict[str, str] = {}
        for task_id, task in problems.items():
            libs = [str(lib) for lib in (task.get("libs") or [])]
            if not libs:
                task_type_by_id[task_id] = "stdlib"
                continue
            task_type_by_id[task_id] = min(libs, key=lambda lib: (freq[lib], lib))
        return task_type_by_id

    def set_phase(self, phase: str) -> None:
        """Switch the active task pool among 'train', 'val', and 'test'.

        The BCB train/val/test split comes from one dataset via
        `split_dataset` (unlike ALFWorld's separate seen/unseen game
        files), so a single adapter instance serves all three phases --
        the caller (run_bcb.py) flips this before/after a validation or
        frozen-test pass instead of constructing a second adapter.
        """
        if phase not in self._PHASES:
            raise ValueError(f"Unknown BCB phase: {phase!r} (expected one of {self._PHASES})")
        self._phase = phase

    def reset_epoch_tracking(self) -> None:
        """Clear the seen-ids set for the CURRENT non-train phase.

        Call once, right after set_phase(), at the start of each
        validation or frozen-test pass -- so a fixed batch_size that
        doesn't evenly divide the pool wraps around and re-dispatches a
        few already-seen tasks to fill the last batch; those slots are
        marked duplicate (see reset()) so EpisodeRunner excludes them
        from aggregated eval metrics instead of double-counting them.
        No-op for the train phase, which is never duplicate-tracked (it
        cycles forever by design).
        """
        if self._phase in self._seen_ids:
            self._seen_ids[self._phase] = set()

    def num_val_tasks(self) -> int:
        """Total distinct tasks in the held-out val split (loads data if not loaded yet)."""
        self._ensure_loaded()
        return len(self._val_ids)

    def num_test_tasks(self) -> int:
        """Total distinct tasks in the frozen held-out test split (loads data if not loaded yet)."""
        self._ensure_loaded()
        return len(self._test_ids)

    def reset(self, **kwargs: Any) -> EpisodeResetResult:
        self._ensure_loaded()
        phase = self._phase
        pool = self._pool_for(phase)
        observations: List[str] = []
        infos: List[Dict[str, Any]] = []
        task_ids: List[Optional[str]] = []

        if not pool:
            self._last_reset_task_ids = []
            return EpisodeResetResult(observations=[], infos=[])

        for _ in range(self._batch_size):
            task_id = pool[self._cursors[phase] % len(pool)]
            self._cursors[phase] += 1
            is_duplicate = False
            seen = self._seen_ids.get(phase)
            if seen is not None:
                is_duplicate = task_id in seen
                if not is_duplicate:
                    seen.add(task_id)
            task = self._problems[task_id]
            prompt = get_prompt(task, split=self._split)
            task_ids.append(task_id)
            observations.append(prompt)
            infos.append(
                {
                    "task_id": task_id,
                    "episode_id": task_id,
                    "task_description": prompt,
                    "task_type": self._task_type_by_id.get(task_id, "stdlib"),
                    "entry_point": task.get("entry_point", ""),
                    "libs": list(task.get("libs") or []),
                    "phase": self._phase,
                    "duplicate": is_duplicate,
                }
            )

        self._last_reset_task_ids = task_ids
        self._log_reset_output(observations, infos)
        return EpisodeResetResult(observations=observations, infos=infos)

    def step(self, actions: List[Any], **kwargs: Any) -> EpisodeStepResult:
        observations: List[str] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []

        for idx, action in enumerate(actions):
            task_id = (
                self._last_reset_task_ids[idx]
                if idx < len(self._last_reset_task_ids)
                else None
            )
            if task_id is None:
                observations.append("")
                rewards.append(0.0)
                dones.append(True)
                infos.append({"status": "NO_TASK", "error": "no task for this slot"})
                continue

            task = self._problems[task_id]
            raw_response = str(action or "")
            code = extract_code_from_response(raw_response)
            eval_result = self._evaluate(task=task, code=code)
            status = eval_result.get("status")
            reward = 1.0 if status == "PASS" else 0.0

            observations.append(f"Evaluation result: {status}")
            rewards.append(reward)
            dones.append(True)
            infos.append(
                {
                    "task_id": task_id,
                    "episode_id": task_id,
                    "task_type": self._task_type_by_id.get(task_id, "stdlib"),
                    "phase": self._phase,
                    "status": status,
                    "error": eval_result.get("error"),
                    "code": code,
                    "raw_response": raw_response,
                }
            )

        self._log_step_output(actions, observations, rewards, dones, infos)
        return EpisodeStepResult(observations=observations, rewards=rewards, dones=dones, infos=infos)

    def _evaluate(self, *, task: Dict[str, Any], code: str) -> Dict[str, Any]:
        """Evaluate one solution via the official BigCodeBench untrusted_check
        harness. Same compile-check -> sanitize -> untrusted_check ->
        status-mapping sequence the old BCBRunner._evaluate_one used.
        """
        task_id = str(task.get("task_id", "unknown"))
        entry_point = str(task.get("entry_point", "task_func"))
        test_code = str(task.get("test", "") or "")

        if not test_code:
            return {"task_id": task_id, "status": "SYNTAX_OK", "error": "no_test_code"}

        try:
            compile(code, "<string>", "exec")
        except SyntaxError as exc:
            return {"task_id": task_id, "status": "SYNTAX_ERROR", "error": str(exc)}

        clean_code = sanitize_code(code, entry_point, bcb_repo=self._bcb_repo)

        from bigcodebench.eval import FAIL, PASS, TIMEOUT  # type: ignore

        stat, details, err, hard_timed_out = run_untrusted_check_with_hard_timeout(
            code=clean_code,
            test_code=test_code,
            entry_point=entry_point,
            max_as_limit=30 * 1024,
            max_data_limit=30 * 1024,
            max_stack_limit=10,
            min_time_limit=1.0,
            gt_time_limit=self._eval_timeout_s,
            hard_timeout_s=self._untrusted_hard_timeout_s,
            bcb_repo=self._bcb_repo,
        )

        if hard_timed_out:
            return {"task_id": task_id, "status": "TIMEOUT", "error": err or "hard_timeout"}
        if err:
            return {"task_id": task_id, "status": "RUNTIME_ERROR", "error": err}
        if stat == PASS:
            return {"task_id": task_id, "status": "PASS"}
        if stat == TIMEOUT:
            return {"task_id": task_id, "status": "TIMEOUT", "error": "timeout"}
        if stat == FAIL:
            return {"task_id": task_id, "status": "FAIL", "error": str(details)[:500] if details else "fail"}
        return {"task_id": task_id, "status": "UNKNOWN", "error": str(stat)}

    def close(self) -> None:
        return

    def episode_id(self, index: int = 0) -> Optional[str]:
        if index >= len(self._last_reset_task_ids):
            return None
        return self._last_reset_task_ids[index]

    def task_type(self, index: int = 0) -> Optional[str]:
        if index >= len(self._last_reset_task_ids):
            return None
        task_id = self._last_reset_task_ids[index]
        if task_id is None:
            return None
        return self._task_type_by_id.get(task_id)

    def is_batch(self) -> bool:
        return self._batch_size > 1

    def known_task_types(self) -> List[str]:
        self._ensure_loaded()
        return sorted(set(self._task_type_by_id.values()))

    def num_tasks(self) -> int:
        """Total distinct tasks in the train split (loads data if not loaded yet)."""
        self._ensure_loaded()
        return len(self._train_ids)

    def _log_reset_output(self, observations: List[str], infos: List[Dict[str, Any]]) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        for idx, observation in enumerate(observations):
            info = infos[idx] if idx < len(infos) else {}
            logger.info(
                "BCB reset[%s]: task_id=%s task_type=%s prompt=%s",
                idx,
                info.get("task_id"),
                info.get("task_type"),
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
                "BCB step[%s]: task_id=%s status=%s reward=%.3f done=%s",
                idx,
                info.get("task_id"),
                info.get("status"),
                float(reward),
                bool(done),
            )


__all__ = [
    "BCBEpisodeEnvAdapter",
    "EpisodeResetResult",
    "EpisodeStepResult",
]
