"""Train DepthCast on a real FI-2010 training file.

Usage:
    python train.py /path/to/Train_Dst_NoAuction_DecPre_CF_7.txt

The script loads the config next to it, reads the FI-2010 matrix, holds
out the last 20 percent of events for validation, and trains.
"""

from __future__ import annotations

import sys
from pathlib import Path

from depthcast.config import ExperimentConfig
from depthcast.data.fi2010 import load_lob_matrix
from depthcast.logging_utils import configure_logging, get_logger
from depthcast.pipeline import make_loader, matrix_to_dataset, split_session
from depthcast.training import Trainer

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def main(train_file: str) -> None:
    """Train a model on the given FI-2010 training file."""
    config = ExperimentConfig.from_yaml(_CONFIG_PATH)
    configure_logging(config.train.log_level)

    matrix = load_lob_matrix(train_file)
    train_matrix, val_matrix = split_session(matrix, val_fraction=0.2)
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
    best = min(history, key=lambda report: report.val_loss)
    logger.info(
        "Best epoch %d: val_loss=%.4f val_acc=%.4f",
        best.epoch,
        best.val_loss,
        best.val_accuracy,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python train.py <fi2010_train_file>")
        raise SystemExit(2)
    main(sys.argv[1])
