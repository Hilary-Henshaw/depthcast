"""Tests for the training loop, including early stopping."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from depthcast.config import ExperimentConfig
from depthcast.pipeline import make_loader, matrix_to_dataset, split_session
from depthcast.training.trainer import Trainer

_Batch = tuple[torch.Tensor, torch.Tensor]


def _loaders(
    config: ExperimentConfig, matrix: NDArray[np.float64]
) -> tuple[DataLoader[_Batch], DataLoader[_Batch]]:
    train_matrix, val_matrix = split_session(matrix, 0.25)
    train = make_loader(
        matrix_to_dataset(train_matrix, config.data),
        config.data,
        shuffle=True,
    )
    val = make_loader(
        matrix_to_dataset(val_matrix, config.data),
        config.data,
        shuffle=False,
    )
    return train, val


@pytest.mark.integration
def test_fit_runs_and_checkpoints_best(
    fast_config: ExperimentConfig,
    synthetic_matrix: NDArray[np.float64],
) -> None:
    train_loader, val_loader = _loaders(fast_config, synthetic_matrix)
    trainer = Trainer.build(fast_config)
    history = trainer.fit(train_loader, val_loader)

    assert len(history) == 1
    report = history[0]
    assert 0.0 <= report.val_accuracy <= 1.0
    assert report.is_best is True
    assert trainer.best_checkpoint_path is not None
    assert trainer.best_checkpoint_path.is_file()


@pytest.mark.integration
def test_grad_clip_path_executes(
    fast_config: ExperimentConfig,
    synthetic_matrix: NDArray[np.float64],
) -> None:
    config = fast_config.model_copy(deep=True)
    config.train.grad_clip = 1.0
    train_loader, val_loader = _loaders(config, synthetic_matrix)
    trainer = Trainer.build(config)
    history = trainer.fit(train_loader, val_loader)
    assert len(history) == 1


def test_build_is_reproducible(fast_config: ExperimentConfig) -> None:
    first = Trainer.build(fast_config)
    second = Trainer.build(fast_config)
    param_a = next(first.model.parameters())
    param_b = next(second.model.parameters())
    assert torch.equal(param_a, param_b)


def test_early_stopping_halts_without_improvement(
    fast_config: ExperimentConfig,
    synthetic_matrix: NDArray[np.float64],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = fast_config.model_copy(deep=True)
    config.train.epochs = 5
    config.train.early_stop_patience = 1
    train_loader, val_loader = _loaders(config, synthetic_matrix)
    trainer = Trainer.build(config)

    # Replace the heavy passes with constant stubs so the test is fast
    # and the validation loss never improves after the first epoch.
    monkeypatch.setattr(trainer, "_train_one_epoch", lambda loader: 0.5)
    monkeypatch.setattr(trainer, "_evaluate", lambda loader: (1.0, 0.33))

    history = trainer.fit(train_loader, val_loader)
    assert len(history) == 2
    assert history[0].is_best is True
    assert history[1].is_best is False
