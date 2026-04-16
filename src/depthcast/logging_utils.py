"""Structured logging helpers for DepthCast.

The library never writes to ``stdout`` directly. Every module obtains a
logger through :func:`get_logger` and emits records with contextual
fields, while the command-line entry point calls
:func:`configure_logging` once to attach a single formatted handler.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_ROOT_NAME: Final[str] = "depthcast"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Attach a single stream handler to the DepthCast root logger.

    Calling this more than once replaces the existing handlers so that
    repeated CLI invocations inside one process do not duplicate output.

    Args:
        level: A standard logging level name or numeric value.

    Raises:
        ValueError: If ``level`` is a string that is not a known level.
    """
    resolved = _resolve_level(level)
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(resolved)
    root.propagate = False

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(resolved)
    handler.setFormatter(
        logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    )

    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child of the DepthCast root logger.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A logger whose name is rooted under ``depthcast`` so that a
        single :func:`configure_logging` call governs all output.
    """
    if name == "__main__" or not name.startswith(_ROOT_NAME):
        return logging.getLogger(f"{_ROOT_NAME}.{name}")
    return logging.getLogger(name)


def _resolve_level(level: int | str) -> int:
    """Normalize a level name or number into a logging level int."""
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(level.upper())
    if not isinstance(resolved, int):
        raise ValueError(f"Unknown logging level: {level!r}")
    return resolved
