"""Shared fixtures for the DepthCast test suite.

Fixtures favour tiny, fast objects so the whole suite runs on CPU in a
few seconds without the multi-gigabyte FI-2010 archive. A short window
and small synthetic sessions keep the convolutional stack inexpensive
while still exercising every shape transition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from depthcast.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from depthcast.data.synthetic import generate_lob_matrix

# Keep the whole suite single-threaded so it never saturates a laptop.
torch.set_num_threads(1)

# A window comfortably larger than the 18 steps the convolutions consume,
# but small enough to keep tests fast.
FAST_WINDOW = 50


@pytest.fixture(scope="session")
def synthetic_matrix() -> NDArray[np.float64]:
    """A small, deterministic FI-2010-shaped session matrix."""
    return generate_lob_matrix(num_events=800, seed=7, move_threshold=0.0005)


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    """A four-channel model that trains in well under a second."""
    return ModelConfig(conv_channels=4, fusion_channels=4, lstm_hidden=8)


@pytest.fixture
def fast_config(
    tmp_path: Path, tiny_model_config: ModelConfig
) -> ExperimentConfig:
    """A minimal experiment configuration for quick training runs."""
    return ExperimentConfig(
        data=DataConfig(window=FAST_WINDOW, batch_size=32),
        model=tiny_model_config,
        train=TrainConfig(
            epochs=1,
            early_stop_patience=None,
            device="cpu",
            checkpoint_dir=tmp_path / "checkpoints",
        ),
    )
