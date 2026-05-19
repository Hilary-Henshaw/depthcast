"""Compare validation accuracy across prediction horizons."""

from __future__ import annotations

from pathlib import Path

import torch

from depthcast.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from depthcast.constants import LABEL_HORIZONS
from depthcast.data import generate_lob_matrix
from depthcast.logging_utils import configure_logging, get_logger
from depthcast.pipeline import make_loader, matrix_to_dataset, split_session
from depthcast.training import Trainer

logger = get_logger(__name__)


def _config(horizon_index: int, checkpoint_dir: Path) -> ExperimentConfig:
    """Build a tiny config targeting one horizon."""
    return ExperimentConfig(
        data=DataConfig(window=50, batch_size=32, horizon_index=horizon_index),
        model=ModelConfig(conv_channels=8, fusion_channels=8, lstm_hidden=16),
        train=TrainConfig(
            epochs=2,
            early_stop_patience=None,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        ),
    )


def main() -> None:
    """Train one model per horizon and report validation accuracy."""
    configure_logging()
    torch.set_num_threads(1)

    matrix = generate_lob_matrix(
        num_events=4000, seed=2, move_threshold=0.0005
    )
    train_matrix, val_matrix = split_session(matrix, val_fraction=0.25)

    results: dict[int, float] = {}
    for index in range(len(LABEL_HORIZONS)):
        config = _config(index, Path("checkpoints") / f"h{index}")
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
        history = trainer.fit(train_loader, val_loader)
        best = max(history, key=lambda report: report.val_accuracy)
        results[config.data.horizon_ticks] = best.val_accuracy

    logger.info("Validation accuracy by horizon (events ahead):")
    for horizon, accuracy in sorted(results.items()):
        logger.info("  horizon %3d: accuracy %.4f", horizon, accuracy)


if __name__ == "__main__":
    main()
