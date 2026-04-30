"""Portable, weights-only checkpoints for DepthCast models.

The reference notebook pickles the entire model object, which couples the
saved file to the exact class definition and is unsafe to load from an
untrusted source. DepthCast instead persists the state dictionary plus a
JSON copy of the configuration, so a checkpoint can be reloaded with
PyTorch's safe ``weights_only`` path and the model rebuilt from its
recorded architecture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict

from depthcast.config import ExperimentConfig
from depthcast.logging_utils import get_logger
from depthcast.models.network import DepthCastNet

logger = get_logger(__name__)

_FORMAT_VERSION = 1


class Checkpoint(BaseModel):
    """A loaded checkpoint with its model, config, and metadata."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        protected_namespaces=(),
    )

    model: DepthCastNet
    config: ExperimentConfig
    epoch: int
    val_loss: float


def save_checkpoint(
    path: str | Path,
    model: DepthCastNet,
    config: ExperimentConfig,
    epoch: int,
    val_loss: float,
) -> Path:
    """Persist a model's weights alongside its configuration.

    Args:
        path: Destination file path; parent directories are created.
        model: The trained network whose weights to save.
        config: The experiment configuration that produced the model.
        epoch: The epoch index this checkpoint corresponds to.
        val_loss: The validation loss recorded for this checkpoint.

    Returns:
        The path that was written.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "model_state": model.state_dict(),
        "config_json": json.dumps(config.model_dump(mode="json")),
        "epoch": int(epoch),
        "val_loss": float(val_loss),
    }
    torch.save(payload, out_path)
    logger.info("Saved checkpoint to %s (val_loss=%.4f)", out_path, val_loss)
    return out_path


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> Checkpoint:
    """Reconstruct a model and configuration from a checkpoint file.

    Args:
        path: Path to a file written by :func:`save_checkpoint`.
        map_location: Device onto which tensors are loaded.

    Returns:
        A :class:`Checkpoint` with an evaluation-ready model.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the checkpoint format version is unsupported.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(
        ckpt_path, map_location=map_location, weights_only=True
    )
    version = payload.get("format_version")
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {version!r}; "
            f"expected {_FORMAT_VERSION}."
        )

    config = ExperimentConfig.model_validate(
        json.loads(payload["config_json"])
    )
    model = DepthCastNet(config.model)
    model.load_state_dict(payload["model_state"])
    model.eval()
    logger.info("Loaded checkpoint from %s", ckpt_path)
    return Checkpoint(
        model=model,
        config=config,
        epoch=int(payload["epoch"]),
        val_loss=float(payload["val_loss"]),
    )
