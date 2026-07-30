"""Epoch-boundary checkpoint save/resume for the EpisodeRunner-based train scripts.

Only epoch boundaries are checkpointed, not mid-epoch section positions:
the env adapters (AlfWorldEpisodeEnvAdapter/BCBEpisodeEnvAdapter) track
dataset-cycling position (which games/tasks have been dispatched this
epoch) purely in-memory with no serializable cursor, so there is no way to
resume mid-epoch without re-dispatching (and re-spending LLM calls on)
already-completed sections. Resuming from a completed epoch boundary is
safe and loses no progress; a run configured with num_epochs=1 gets no
within-run restart safety from this alone, since it never crosses an
epoch boundary.

The skill graph itself needs no separate snapshot here -- MemoryService
persists every mutation straight to its sqlite db_path -- so resuming just
means pointing a fresh EpisodeRunner at the same db_path plus restoring its
own progress/telemetry counters via get_checkpoint_state()/load_checkpoint_state().
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def save_epoch_checkpoint(run_dir: Path, epoch_idx: int, runner: Any, db_path: Path) -> Path:
    """Write the checkpoint for a just-completed epoch (0-based epoch_idx).

    Writes checkpoints/epoch_{idx:04d}.json plus an overwritten
    checkpoints/latest.json pointer, so a resume path can name either the
    run_dir itself (resumes from the latest epoch) or a specific epoch
    file (resumes from exactly that one).
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch_idx),
        "db_path": str(db_path),
        "run_id": getattr(runner, "run_id", None),
        "saved_at": datetime.now().isoformat(),
        "runner_state": runner.get_checkpoint_state(),
    }
    epoch_path = ckpt_dir / f"epoch_{epoch_idx:04d}.json"
    with open(epoch_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    latest_path = ckpt_dir / "latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Saved checkpoint for epoch %s to %s", epoch_idx, epoch_path)
    return epoch_path


def load_checkpoint(resume_path: str, resume_epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Resolve and load a checkpoint from a resume_path that may be:
    - a direct path to a checkpoint JSON file,
    - a "checkpoints" directory,
    - a run_dir containing a "checkpoints" subdirectory.

    If resume_epoch is given, loads that specific epoch's file; otherwise
    loads checkpoints/latest.json. Returns None (and logs a warning)
    instead of raising if nothing resolvable is found -- a missing/bad
    resume path should not by itself crash the run.
    """
    base = Path(resume_path)
    candidates = []
    if base.is_file():
        candidates.append(base)
    else:
        ckpt_dir = base if base.name == "checkpoints" else base / "checkpoints"
        if resume_epoch is not None:
            candidates.append(ckpt_dir / f"epoch_{int(resume_epoch):04d}.json")
        candidates.append(ckpt_dir / "latest.json")

    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    state = json.load(f)
                logger.info("Resuming from checkpoint %s (epoch %s)", candidate, state.get("epoch"))
                return state
            except Exception:
                logger.warning("Failed to load checkpoint from %s", candidate, exc_info=True)
                return None

    logger.warning("ckpt_resume_enabled but no checkpoint found under %s", resume_path)
    return None


__all__ = ["save_epoch_checkpoint", "load_checkpoint"]
