"""High-level glue that turns raw matrices into ready-to-train loaders.

These helpers sit between the low-level data readers and the trainer so
that the command-line interface and the example scripts share one tested
path: split a session along time, window each split, and wrap it in a
loader.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from depthcast.config import DataConfig
from depthcast.data.dataset import LobWindowDataset, build_dataloader
from depthcast.data.fi2010 import split_features_labels
from depthcast.logging_utils import get_logger

logger = get_logger(__name__)

# Windows leave the dataset as tensors, so loaders yield tensor batches.
Batch = tuple[torch.Tensor, torch.Tensor]


def split_session(
    matrix: NDArray[np.float64], val_fraction: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split a session matrix into train and validation along time.

    Args:
        matrix: A ``(rows, events)`` session matrix.
        val_fraction: Fraction of trailing events held out for
            validation, in the open interval ``(0, 1)``.

    Returns:
        A pair ``(train_matrix, val_matrix)`` split on the event axis.

    Raises:
        ValueError: If ``val_fraction`` is not in ``(0, 1)`` or a split
            would be empty.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be in (0, 1); got {val_fraction}."
        )
    num_events = matrix.shape[1]
    cut = int(np.floor(num_events * (1.0 - val_fraction)))
    if cut <= 0 or cut >= num_events:
        raise ValueError(
            "val_fraction leaves an empty split for a session of "
            f"{num_events} events."
        )
    logger.debug("Splitting %d events at index %d", num_events, cut)
    return matrix[:, :cut], matrix[:, cut:]


def matrix_to_dataset(
    matrix: NDArray[np.float64], data: DataConfig
) -> LobWindowDataset:
    """Window a raw session matrix into a supervised dataset.

    Args:
        matrix: A ``(rows, events)`` session matrix.
        data: Data configuration describing window and horizon.

    Returns:
        A windowed dataset over the session.
    """
    features, labels = split_features_labels(
        matrix,
        horizon_index=data.horizon_index,
        num_features=data.num_features,
    )
    return LobWindowDataset(features, labels, window=data.window)


def make_loader(
    dataset: LobWindowDataset, data: DataConfig, *, shuffle: bool
) -> DataLoader[Batch]:
    """Build a loader for a dataset using the data configuration.

    Args:
        dataset: The windowed dataset to iterate.
        data: Data configuration carrying batch size and workers.
        shuffle: Whether to shuffle batches each epoch.

    Returns:
        A configured data loader.
    """
    return build_dataloader(
        dataset,
        batch_size=data.batch_size,
        shuffle=shuffle,
        num_workers=data.num_workers,
    )
