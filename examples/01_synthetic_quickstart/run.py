"""End-to-end DepthCast run on synthetic data.

Generates a synthetic session, trains a small model for two epochs, and
prints an evaluation report. Intended as a fast smoke test of the whole
pipeline on a CPU.
"""

from __future__ import annotations

from pathlib import Path

import torch

from depthcast.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from depthcast.data import generate_lob_matrix
from depthcast.evaluation import evaluate_classifier
from depthcast.logging_utils import configure_logging, get_logger
from depthcast.pipeline import make_loader, matrix_to_dataset, split_session
from depthcast.runtime import resolve_device
from depthcast.training import Trainer, load_checkpoint

logger = get_logger(__name__)


def main() -> None:
    """Train and evaluate a tiny model on synthetic data."""
    configure_logging()
    torch.set_num_threads(1)

    config = ExperimentConfig(
        data=DataConfig(window=50, batch_size=32),
        model=ModelConfig(conv_channels=8, fusion_channels=8, lstm_hidden=16),
        train=TrainConfig(
            epochs=2,
            early_stop_patience=None,
            device="cpu",
            checkpoint_dir=Path("checkpoints"),
        ),
    )

    matrix = generate_lob_matrix(
        num_events=4000, seed=0, move_threshold=0.0005
    )
    train_matrix, val_matrix = split_session(matrix, val_fraction=0.25)
    train_loader = make_loader(
        matrix_to_dataset(train_matrix, config.data),
        config.data,
        shuffle=True,
    )
    val_loader = make_loader(
        matrix_to_dataset(val_matrix, config.data),
        config.data,
        shuffle=False,
    )

    trainer = Trainer.build(config)
    trainer.fit(train_loader, val_loader)

    assert trainer.best_checkpoint_path is not None
    best = load_checkpoint(trainer.best_checkpoint_path)
    metrics = evaluate_classifier(
        best.model, val_loader, resolve_device("cpu")
    )

    logger.info("Final accuracy: %.4f", metrics.accuracy)
    logger.info("Final macro F1: %.4f", metrics.macro_f1)
    logger.info("Per-class F1: %s", metrics.per_class_f1)
    logger.info("Classification report:\n%s", metrics.report)


if __name__ == "__main__":
    main()
