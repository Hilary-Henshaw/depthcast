"""Save a checkpoint, reload it, and run single-window inference."""

from __future__ import annotations

from pathlib import Path

import torch

from depthcast.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from depthcast.constants import MOVEMENT_CLASSES
from depthcast.data import generate_lob_matrix
from depthcast.logging_utils import configure_logging, get_logger
from depthcast.pipeline import make_loader, matrix_to_dataset, split_session
from depthcast.training import Trainer, load_checkpoint

logger = get_logger(__name__)


def main() -> None:
    """Train, persist, reload, and predict on one window."""
    configure_logging()
    torch.set_num_threads(1)

    config = ExperimentConfig(
        data=DataConfig(window=50, batch_size=32),
        model=ModelConfig(conv_channels=8, fusion_channels=8, lstm_hidden=16),
        train=TrainConfig(
            epochs=1,
            early_stop_patience=None,
            device="cpu",
            checkpoint_dir=Path("checkpoints"),
        ),
    )

    matrix = generate_lob_matrix(
        num_events=3000, seed=1, move_threshold=0.0005
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

    loaded = load_checkpoint(trainer.best_checkpoint_path)
    logger.info(
        "Reloaded checkpoint from epoch %d (val_loss=%.4f)",
        loaded.epoch,
        loaded.val_loss,
    )
    logger.info("Recorded window size: %d", loaded.config.data.window)

    val_dataset = matrix_to_dataset(val_matrix, loaded.config.data)
    window, true_label = val_dataset[0]
    probabilities = loaded.model.predict_proba(window.unsqueeze(0))
    distribution = {
        name: round(float(prob), 4)
        for name, prob in zip(
            MOVEMENT_CLASSES, probabilities.squeeze(0), strict=True
        )
    }
    logger.info("True class: %s", MOVEMENT_CLASSES[int(true_label)])
    logger.info("Predicted distribution: %s", distribution)


if __name__ == "__main__":
    main()
