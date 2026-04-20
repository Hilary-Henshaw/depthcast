"""Readers for the FI-2010 limit-order-book benchmark.

The FI-2010 distribution stores each session as a dense text matrix whose
rows are features and whose columns are successive order-book events. The
first :data:`NUM_LOB_FEATURES` rows are the normalised depth snapshot and
the final rows hold one direction label per prediction horizon. These
helpers turn that layout into the ``(time, feature)`` arrays the rest of
the pipeline expects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from depthcast.constants import (
    LABEL_HORIZONS,
    NUM_LOB_FEATURES,
    RAW_LABEL_OFFSET,
)
from depthcast.logging_utils import get_logger

logger = get_logger(__name__)

_NUM_LABEL_ROWS = len(LABEL_HORIZONS)


def load_lob_matrix(path: str | Path) -> NDArray[np.float64]:
    """Load a raw FI-2010 session matrix from a whitespace text file.

    Args:
        path: Path to a ``Train_*`` or ``Test_*`` FI-2010 text file.

    Returns:
        A two-dimensional array shaped ``(rows, events)`` exactly as the
        file is laid out, with features on the row axis.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not parse as a 2-D matrix or has
            too few rows to contain features and every horizon label.
    """
    matrix_path = Path(path)
    if not matrix_path.is_file():
        raise FileNotFoundError(f"FI-2010 file not found: {matrix_path}")

    logger.info("Loading FI-2010 matrix from %s", matrix_path)
    matrix = np.loadtxt(matrix_path)
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2-D matrix, got {matrix.ndim} dimensions."
        )

    min_rows = NUM_LOB_FEATURES + _NUM_LABEL_ROWS
    if matrix.shape[0] < min_rows:
        raise ValueError(
            f"Matrix needs at least {min_rows} rows "
            f"({NUM_LOB_FEATURES} features + {_NUM_LABEL_ROWS} labels); "
            f"got {matrix.shape[0]}."
        )
    logger.debug("Loaded matrix with shape %s", matrix.shape)
    return matrix


def split_features_labels(
    matrix: NDArray[np.float64],
    horizon_index: int,
    num_features: int = NUM_LOB_FEATURES,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Separate a session matrix into time-major features and labels.

    Args:
        matrix: A ``(rows, events)`` matrix from :func:`load_lob_matrix`.
        horizon_index: Index into :data:`LABEL_HORIZONS` selecting which
            prediction horizon to return labels for.
        num_features: Number of leading rows treated as features.

    Returns:
        A pair ``(features, labels)`` where ``features`` has shape
        ``(events, num_features)`` as ``float32`` and ``labels`` has
        shape ``(events,)`` as zero-based ``int64`` class indices.

    Raises:
        IndexError: If ``horizon_index`` is out of range.
        ValueError: If decoded labels fall outside the expected classes.
    """
    if not 0 <= horizon_index < _NUM_LABEL_ROWS:
        raise IndexError(
            f"horizon_index must be in [0, {_NUM_LABEL_ROWS - 1}]; "
            f"got {horizon_index}."
        )

    features = matrix[:num_features, :].T.astype(np.float32, copy=False)

    label_block = matrix[-_NUM_LABEL_ROWS:, :]
    raw_labels = label_block[horizon_index, :]
    labels = raw_labels.astype(np.int64) - RAW_LABEL_OFFSET

    _validate_labels(labels)
    logger.debug(
        "Split matrix into features %s and labels %s",
        features.shape,
        labels.shape,
    )
    return features, labels


def _validate_labels(labels: NDArray[np.int64]) -> None:
    """Ensure decoded labels are valid zero-based class indices."""
    from depthcast.constants import NUM_CLASSES

    if labels.size == 0:
        raise ValueError("Label vector is empty.")
    low = int(labels.min())
    high = int(labels.max())
    if low < 0 or high >= NUM_CLASSES:
        raise ValueError(
            "Decoded labels must lie in "
            f"[0, {NUM_CLASSES - 1}]; got range [{low}, {high}]. "
            "Check that the file follows the FI-2010 label convention."
        )
