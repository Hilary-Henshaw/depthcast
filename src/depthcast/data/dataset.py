"""A memory-frugal sliding-window dataset over order-book snapshots.

The original reference implementation expands every window into a dense
``(samples, window, features)`` tensor, duplicating each snapshot once per
window. :class:`LobWindowDataset` instead holds the contiguous feature
matrix and produces a window on access, which keeps memory proportional
to the raw session rather than to the number of windows.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from depthcast.data.windowing import count_windows, label_index_for_window
from depthcast.logging_utils import get_logger

logger = get_logger(__name__)


class LobWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazily windowed view over a single order-book session.

    Each item is a ``(features, label)`` pair where ``features`` has
    shape ``(1, window, num_features)`` so it can feed a single-channel
    2-D convolution directly, and ``label`` is a scalar class index.
    """

    def __init__(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        window: int,
    ) -> None:
        """Wrap feature and label arrays in a windowed dataset.

        Args:
            features: Array shaped ``(num_events, num_features)``.
            labels: Array shaped ``(num_events,)`` of class indices.
            window: Number of snapshots per sample.

        Raises:
            ValueError: If shapes disagree, ``window`` is not positive,
                or the session is shorter than one window.
        """
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D, got {features.ndim} dimensions."
            )
        if labels.ndim != 1:
            raise ValueError(
                f"labels must be 1-D, got {labels.ndim} dimensions."
            )
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                "features and labels must share the event axis; "
                f"got {features.shape[0]} and {labels.shape[0]}."
            )

        self._window = window
        self._length = count_windows(features.shape[0], window)
        if self._length == 0:
            raise ValueError(
                f"Session of {features.shape[0]} events is shorter than "
                f"the window of {window}."
            )

        self._features = torch.from_numpy(
            np.ascontiguousarray(features, dtype=np.float32)
        )
        self._labels = torch.from_numpy(
            np.ascontiguousarray(labels, dtype=np.int64)
        )
        logger.debug(
            "Built dataset with %d windows of length %d",
            self._length,
            window,
        )

    @property
    def num_features(self) -> int:
        """Number of feature columns per snapshot."""
        return int(self._features.shape[1])

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(
                f"index {index} out of range for {self._length} windows."
            )
        stop = index + self._window
        window = self._features[index:stop].unsqueeze(0)
        label = self._labels[label_index_for_window(index, self._window)]
        return window, label


def build_dataloader(
    dataset: LobWindowDataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a :class:`~torch.utils.data.DataLoader` for a dataset.

    Args:
        dataset: The windowed dataset to iterate.
        batch_size: Number of windows per batch.
        shuffle: Whether to shuffle window order each epoch.
        num_workers: Worker processes for loading; ``0`` loads inline.

    Returns:
        A configured data loader.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
