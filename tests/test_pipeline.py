"""Tests for the data-orchestration helpers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from depthcast.config import DataConfig
from depthcast.data.windowing import count_windows
from depthcast.pipeline import make_loader, matrix_to_dataset, split_session


def test_split_session_divides_event_axis(
    synthetic_matrix: NDArray[np.float64],
) -> None:
    train, val = split_session(synthetic_matrix, 0.2)
    assert train.shape[0] == synthetic_matrix.shape[0]
    assert train.shape[1] + val.shape[1] == synthetic_matrix.shape[1]
    assert val.shape[1] == pytest.approx(
        synthetic_matrix.shape[1] * 0.2, abs=1
    )


def test_split_session_rejects_out_of_range_fraction(
    synthetic_matrix: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        split_session(synthetic_matrix, 1.5)


def test_split_session_rejects_empty_split() -> None:
    matrix = np.ones((45, 10))
    with pytest.raises(ValueError, match="empty split"):
        split_session(matrix, 0.999)


def test_matrix_to_dataset_window_count(
    synthetic_matrix: NDArray[np.float64],
) -> None:
    data = DataConfig(window=50)
    dataset = matrix_to_dataset(synthetic_matrix, data)
    expected = count_windows(synthetic_matrix.shape[1], 50)
    assert len(dataset) == expected


def test_make_loader_yields_expected_batch(
    synthetic_matrix: NDArray[np.float64],
) -> None:
    data = DataConfig(window=50, batch_size=8)
    dataset = matrix_to_dataset(synthetic_matrix, data)
    loader = make_loader(dataset, data, shuffle=False)
    inputs, targets = next(iter(loader))
    assert inputs.shape == (8, 1, 50, 40)
    assert targets.shape == (8,)
