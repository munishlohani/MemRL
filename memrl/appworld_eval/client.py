"""MemRL-side client for the AppWorld worker subprocess.

Runs inside MemRL's interpreter and must NEVER import `appworld` -- that is
the whole point of the split (see worker.py). Everything here is stdlib plus
JSON over pipes.

One client owns one worker process, and one worker holds at most one live
AppWorld at a time, so a batched adapter keeps one client per slot.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_WORKER_MODULE = "memrl.appworld_eval.worker"


class AppWorldClientError(RuntimeError):
    """Raised when the worker reports an error or the channel breaks."""


class AppWorldClient:
    """Spawn and drive one AppWorld worker process.

    `python_executable` must point at an interpreter with `appworld`
    installed (typically `.venv-appworld/bin/python`); `appworld_root` is the
    data root produced by `appworld download data --root <root>`.
    """

    def __init__(
        self,
        *,
        python_executable: str,
        appworld_root: str,
        experiment_name: str = "memrl",
        project_root: Optional[str] = None,
        startup_timeout: float = 120.0,
    ) -> None:
        self._python = str(python_executable)
        self._root = str(appworld_root)
        self._experiment_name = str(experiment_name)
        # The worker is launched as `-m memrl.appworld_eval.worker`, so the
        # child needs MemRL importable even though it will not import any of
        # MemRL's dependency-heavy modules.
        self._project_root = str(project_root or Path(__file__).resolve().parents[2])
        self._startup_timeout = float(startup_timeout)
        self._proc: Optional[subprocess.Popen] = None
        self._max_interactions: int = 0

    # -- process lifecycle ---------------------------------------------------

    def _ensure_started(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        if not Path(self._python).exists():
            raise AppWorldClientError(
                f"AppWorld interpreter not found: {self._python}. Create it with "
                f"`uv venv .venv-appworld --python 3.12` and "
                f"`uv pip install --python .venv-appworld/bin/python appworld`."
            )
        if not Path(self._root).exists():
            raise AppWorldClientError(
                f"AppWorld data root not found: {self._root}. Download it with "
                f"`appworld download data --root {self._root}`."
            )

        env = dict(os.environ)
        env["APPWORLD_ROOT"] = self._root
        # The worker's venv has its own site-packages; PYTHONPATH only needs to
        # expose MemRL itself so `-m memrl.appworld_eval.worker` resolves.
        env["PYTHONPATH"] = self._project_root + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("PYTHONUNBUFFERED", "1")

        logger.info(
            "Starting AppWorld worker: %s -m %s (root=%s)",
            self._python, _WORKER_MODULE, self._root,
        )
        self._proc = subprocess.Popen(
            [self._python, "-m", _WORKER_MODULE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=self._project_root,
            text=True,
            bufsize=1,
        )

        ready = self._read_response()
        if not ready.get("ready"):
            raise AppWorldClientError(f"worker did not report ready: {ready}")
        return self._proc

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                self._request({"op": "shutdown"}, allow_dead=True)
        except Exception:
            logger.debug("AppWorld worker shutdown request failed", exc_info=True)
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=15)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self._proc = None

    # -- protocol -----------------------------------------------------------

    def _read_response(self) -> Dict[str, Any]:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        line = proc.stdout.readline()
        if not line:
            stderr = ""
            if proc.stderr is not None:
                try:
                    stderr = proc.stderr.read() or ""
                except Exception:
                    stderr = ""
            raise AppWorldClientError(
                "AppWorld worker closed the channel"
                + (f"; stderr tail: {stderr[-800:]}" if stderr.strip() else "")
            )
        try:
            return json.loads(line)
        except Exception as exc:
            raise AppWorldClientError(f"unparseable worker line {line!r}: {exc}") from exc

    def _request(self, payload: Dict[str, Any], *, allow_dead: bool = False) -> Dict[str, Any]:
        proc = self._proc if allow_dead else self._ensure_started()
        if proc is None or proc.stdin is None:
            raise AppWorldClientError("AppWorld worker is not running")
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        response = self._read_response()
        if not response.get("ok", False):
            raise AppWorldClientError(
                f"{response.get('error_type', 'WorkerError')}: {response.get('error')}"
            )
        return response

    # -- operations ---------------------------------------------------------

    def task_ids(self, split: str) -> List[str]:
        return [str(t) for t in self._request({"op": "task_ids", "split": split})["task_ids"]]

    def reset(self, task_id: str) -> Dict[str, Any]:
        response = self._request(
            {
                "op": "reset",
                "task_id": task_id,
                "experiment_name": self._experiment_name,
            }
        )
        self._max_interactions = int(response.get("max_interactions", 0) or 0)
        return response

    def execute(self, code: str) -> Dict[str, Any]:
        return self._request({"op": "execute", "code": code})

    def evaluate(self) -> Dict[str, Any]:
        return self._request({"op": "evaluate"})

    def close_world(self) -> None:
        """Close the active task without tearing down the worker process."""
        if self._proc is not None and self._proc.poll() is None:
            self._request({"op": "close"})

    @property
    def max_interactions(self) -> int:
        return self._max_interactions


__all__ = ["AppWorldClient", "AppWorldClientError"]
