"""Reproducibility and device-selection helpers.

These utilities keep the rest of the codebase free of global state. The
trainer asks :func:`resolve_device` to turn a user preference into a
concrete device and calls :func:`seed_everything` once so that runs are
repeatable.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from depthcast.logging_utils import get_logger

logger = get_logger(__name__)


def resolve_device(choice: str) -> torch.device:
    """Turn a device preference into a concrete :class:`torch.device`.

    Args:
        choice: One of ``"auto"``, ``"cpu"``, or ``"cuda"``. ``"auto"``
            selects CUDA when available and falls back to CPU.

    Returns:
        The resolved device.

    Raises:
        ValueError: If ``choice`` is not a recognised option.
        RuntimeError: If ``"cuda"`` is requested but unavailable.
    """
    normalised = choice.lower()
    if normalised == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug("Auto-selected device: %s", device)
        return torch.device(device)
    if normalised == "cpu":
        return torch.device("cpu")
    if normalised == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available on this host."
            )
        return torch.device("cuda")
    raise ValueError(
        f"Unknown device choice {choice!r}; expected 'auto', 'cpu', or 'cuda'."
    )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    Args:
        seed: Non-negative seed shared across all generators.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative; got {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug("Seeded all generators with %d", seed)
