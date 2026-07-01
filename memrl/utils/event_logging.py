"""Small structured logging helpers used across memory and training flows."""

from __future__ import annotations

import logging
from typing import Any


def _compact_value(value: Any, limit: int = 240) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log a compact structured event in a single line."""
    parts = [event]
    for key in sorted(fields):
        value = fields[key]
        if value is None:
            continue
        parts.append(f"{key}={_compact_value(value)}")
    logger.info(" | ".join(parts))
