"""Tests for portable checkpoint saving and loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from depthcast.config import ExperimentConfig
from depthcast.models.network import DepthCastNet
from depthcast.training.checkpoint import load_checkpoint, save_checkpoint


def test_round_trip_preserves_weights_and_config(
    tmp_path: Path, fast_config: ExperimentConfig
) -> None:
    model = DepthCastNet(fast_config.model)
    path = save_checkpoint(
        tmp_path / "best.pt", model, fast_config, epoch=3, val_loss=0.42
    )
    assert path.is_file()

    loaded = load_checkpoint(path)
    assert loaded.epoch == 3
    assert loaded.val_loss == pytest.approx(0.42)
    assert loaded.config == fast_config

    for original, restored in zip(
        model.state_dict().values(),
        loaded.model.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(original, restored)


def test_loaded_model_is_in_eval_mode(
    tmp_path: Path, fast_config: ExperimentConfig
) -> None:
    model = DepthCastNet(fast_config.model)
    path = save_checkpoint(
        tmp_path / "best.pt", model, fast_config, epoch=1, val_loss=1.0
    )
    loaded = load_checkpoint(path)
    assert loaded.model.training is False


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "absent.pt")


def test_unsupported_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "old.pt"
    torch.save(
        {
            "format_version": 99,
            "model_state": {},
            "config_json": "{}",
            "epoch": 0,
            "val_loss": 0.0,
        },
        path,
    )
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_checkpoint(path)
