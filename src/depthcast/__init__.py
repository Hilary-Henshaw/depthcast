"""DepthCast: limit-order-book price-direction forecasting.

DepthCast trains a convolutional-recurrent network on limit-order-book
snapshots to predict the direction of the next mid-price move. The public
surface re-exports the pieces most callers need: the typed configuration
models, the network, the trainer, and the data utilities.
"""

from __future__ import annotations

from depthcast.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from depthcast.constants import (
    LABEL_HORIZONS,
    MOVEMENT_CLASSES,
    NUM_CLASSES,
    NUM_LOB_FEATURES,
)
from depthcast.models.network import DepthCastNet

__all__ = [
    "LABEL_HORIZONS",
    "MOVEMENT_CLASSES",
    "NUM_CLASSES",
    "NUM_LOB_FEATURES",
    "DataConfig",
    "DepthCastNet",
    "ExperimentConfig",
    "ModelConfig",
    "TrainConfig",
    "__version__",
]

__version__ = "0.1.0"
