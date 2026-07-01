"""Compatibility wrapper for the episode runner base class."""

from __future__ import annotations

from memrl.episode.base import BaseEpisodeRunner


class BaseRunner(BaseEpisodeRunner):
    """Backward-compatible alias kept for the existing benchmark runners."""


__all__ = ["BaseRunner"]
