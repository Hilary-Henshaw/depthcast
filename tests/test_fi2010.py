"""Tests for the FI-2010 reader and label decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from depthcast.constants import LABEL_HORIZONS, NUM_LOB_FEATURES
from depthcast.data.fi2010 import load_lob_matrix, split_features_labels


def _write_matrix(path: Path, matrix: NDArray[np.float64]) -> Path:
    np.savetxt(path, matrix)
    return path


def test_load_round_trips_a_saved_matrix(
    tmp_path: Path, synthetic_matrix: NDArray[np.float64]
) -> None:
    path = _write_matrix(tmp_path / "session.txt", synthetic_matrix)
    loaded = load_lob_matrix(path)
    assert loaded.shape == synthetic_matrix.shape


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_lob_matrix(tmp_path / "nope.txt")


def test_load_rejects_too_few_rows(tmp_path: Path) -> None:
    path = _write_matrix(tmp_path / "tiny.txt", np.ones((10, 50)))
    with pytest.raises(ValueError, match="at least"):
        load_lob_matrix(path)


def test_load_rejects_one_dimensional_file(tmp_path: Path) -> None:
    path = tmp_path / "row.txt"
    path.write_text("1 2 3 4 5\n")
    with pytest.raises(ValueError, match="2-D"):
        load_lob_matrix(path)


def test_split_returns_time_major_features(
    synthetic_matrix: NDArray[np.float64],
) -> None:
    features, labels = split_features_labels(synthetic_matrix, horizon_index=4)
    num_events = synthetic_matrix.shape[1]
    assert features.shape == (num_events, NUM_LOB_FEATURES)
    assert features.dtype == np.float32
    assert labels.shape == (num_events,)
    assert labels.dtype == np.int64


def test_split_produces_zero_based_labels(
    synthetic_matrix: NDArray[np.float64],
) -> None:
    _, labels = split_features_labels(synthetic_matrix, horizon_index=0)
    assert labels.min() >= 0
    assert labels.max() <= 2


def test_split_rejects_out_of_range_horizon(
    synthetic_matrix: NDArray[np.float64],
) -> None:
    with pytest.raises(IndexError, match="horizon_index"):
        split_features_labels(
            synthetic_matrix, horizon_index=len(LABEL_HORIZONS)
        )


def test_split_rejects_invalid_label_values() -> None:
    matrix = np.ones((NUM_LOB_FEATURES + len(LABEL_HORIZONS), 20))
    matrix[-len(LABEL_HORIZONS) :, :] = 9.0
    with pytest.raises(ValueError, match="FI-2010 label convention"):
        split_features_labels(matrix, horizon_index=0)
