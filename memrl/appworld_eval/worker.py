"""AppWorld worker process -- the ONLY module that imports `appworld`.

Why a subprocess at all: AppWorld pins pydantic 1.x and SQLAlchemy 1.4,
while MemRL is built on pydantic 2.x / SQLAlchemy 2.0 (memrl/configs/config.py
uses field_validator/ConfigDict; memrl/service/memory_service.py uses 2.0-style
select()/.mappings()). Installing appworld into MemRL's venv downgrades both
and breaks every benchmark -- verified, not hypothetical. So appworld lives in
its own interpreter (`.venv-appworld`) and this script is run BY that
interpreter, talking to MemRL over a line-delimited JSON protocol on
stdin/stdout.

Why not `appworld serve environment`: AppWorld does support a remote HTTP
mode, but the client side of it is the `AppWorld(remote_environment_url=...)`
constructor -- i.e. it still wants the appworld package in the caller's
interpreter. Speaking its REST schema directly would mean depending on an
undocumented surface that can shift between releases. This protocol is ours,
so it only changes when we change it.

Protocol -- one JSON object per line in, one per line out:

    {"op": "task_ids", "split": "train"}   -> {"ok": true, "task_ids": [...]}
    {"op": "reset", "task_id": "...", "experiment_name": "..."}
                                          -> {"ok": true, "instruction": str,
                                              "max_interactions": int}
    {"op": "execute", "code": "..."}      -> {"ok": true, "output": str,
                                              "task_completed": bool,
                                              "num_interactions": int}
    {"op": "evaluate"}                    -> {"ok": true, "success": bool, ...}
    {"op": "close"}                       -> {"ok": true}
    {"op": "shutdown"}                    -> {"ok": true} then exits

Errors never crash the worker: they come back as
{"ok": false, "error": str, "error_type": str} so one bad task cannot take
down a whole batch.

The worker is long-lived and reused across episodes -- importing appworld and
booting its app code is slow, so `reset` closes the previous world and opens a
new one in the same process rather than paying that cost per episode.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, Optional


def _emit(payload: Dict[str, Any], stream) -> None:
    stream.write(json.dumps(payload, default=str) + "\n")
    stream.flush()


def _value(obj: Any, name: str, default: Any) -> Any:
    """Read `name` off `obj`, calling it when it is a method.

    AppWorld's surface is mixed: `num_interactions`/`max_interactions` are
    attributes but `task_completed` is a METHOD. A plain getattr on the latter
    returns the bound method, and `bool(<bound method>)` is always True -- which
    silently reports every episode as finished after its first step. Resolving
    callables here keeps that trap in one place.
    """
    try:
        attribute = getattr(obj, name)
    except Exception:
        return default
    if callable(attribute):
        try:
            return attribute()
        except Exception:
            return default
    return attribute if attribute is not None else default


class _AppWorldWorker:
    def __init__(self) -> None:
        self._world: Optional[Any] = None
        self._appworld = None
        self._load_task_ids = None

    def _lazy_import(self) -> None:
        """Imported on first use, not at module import, so a bad APPWORLD_ROOT
        surfaces as a protocol error rather than a startup crash with no
        response line."""
        if self._appworld is not None:
            return
        from appworld import AppWorld, load_task_ids  # type: ignore

        self._appworld = AppWorld
        self._load_task_ids = load_task_ids

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        op = str(request.get("op") or "")
        if op == "shutdown":
            self.close()
            return {"ok": True, "shutdown": True}
        if op == "task_ids":
            return self.task_ids(str(request.get("split") or "train"))
        if op == "reset":
            return self.reset(
                task_id=str(request.get("task_id") or ""),
                experiment_name=str(request.get("experiment_name") or "memrl"),
            )
        if op == "execute":
            return self.execute(str(request.get("code") or ""))
        if op == "evaluate":
            return self.evaluate()
        if op == "close":
            self.close()
            return {"ok": True}
        return {"ok": False, "error": f"unknown op: {op!r}", "error_type": "ProtocolError"}

    def task_ids(self, split: str) -> Dict[str, Any]:
        self._lazy_import()
        return {"ok": True, "task_ids": list(self._load_task_ids(split))}

    def reset(self, *, task_id: str, experiment_name: str) -> Dict[str, Any]:
        self._lazy_import()
        self.close()
        world = self._appworld(task_id=task_id, experiment_name=experiment_name)
        world.__enter__()
        self._world = world
        return {
            "ok": True,
            "task_id": task_id,
            "instruction": str(_value(world.task, "instruction", "") or ""),
            "max_interactions": int(_value(world, "max_interactions", 0) or 0),
        }

    def execute(self, code: str) -> Dict[str, Any]:
        if self._world is None:
            return {"ok": False, "error": "no active world; call reset first",
                    "error_type": "StateError"}
        output = self._world.execute(code)
        return {
            "ok": True,
            "output": "" if output is None else str(output),
            "task_completed": bool(_value(self._world, "task_completed", False)),
            "num_interactions": int(_value(self._world, "num_interactions", 0) or 0),
        }

    def evaluate(self) -> Dict[str, Any]:
        if self._world is None:
            return {"ok": False, "error": "no active world; call reset first",
                    "error_type": "StateError"}
        tracker = self._world.evaluate()
        # TestTracker.success is the task-goal-completion bool (all
        # requirements pass). The rest is carried for analysis only -- the
        # adapter maps success alone to reward, matching how the other
        # benchmarks emit 0.0/1.0.
        return {
            "ok": True,
            "success": bool(_value(tracker, "success", False)),
            "pass_count": int(_value(tracker, "pass_count", 0) or 0),
            "fail_count": int(_value(tracker, "fail_count", 0) or 0),
            "num_tests": int(_value(tracker, "num_tests", 0) or 0),
            "pass_percentage": float(_value(tracker, "pass_percentage", 0.0) or 0.0),
            "difficulty": int(_value(tracker, "difficulty", 0) or 0),
            "task_completed": bool(_value(tracker, "task_completed", False)),
        }

    def close(self) -> None:
        world, self._world = self._world, None
        if world is None:
            return
        try:
            world.close()
        except Exception:
            pass


def main() -> None:
    # AppWorld (and the app code it unpacks) prints to stdout. That would be
    # interleaved with protocol lines and corrupt the stream, so claim the real
    # stdout for the protocol and point sys.stdout at stderr -- any stray
    # library print then lands in the worker's log instead of the channel.
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    root = os.environ.get("APPWORLD_ROOT")
    if root:
        try:
            from appworld import update_root  # type: ignore

            update_root(root)
        except Exception:
            # Non-fatal: APPWORLD_ROOT is also read by appworld itself, so a
            # missing update_root in some version should not kill the worker.
            pass

    worker = _AppWorldWorker()
    _emit({"ok": True, "ready": True}, protocol_out)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as exc:
            _emit({"ok": False, "error": f"bad JSON: {exc}",
                   "error_type": "ProtocolError"}, protocol_out)
            continue

        try:
            response = worker.handle(request)
        except Exception as exc:
            response = {
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(limit=8),
            }
        _emit(response, protocol_out)
        if response.get("shutdown"):
            break


if __name__ == "__main__":
    main()
