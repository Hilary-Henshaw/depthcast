"""Tests for the synthetic data generator."""

from __future__ import annotations

import numpy as np
import pytest

from depthcast.constants import LABEL_HORIZONS, NUM_LOB_FEATURES
from depthcast.data.synthetic import generate_lob_matrix


def test_matrix_has_feature_and_label_rows() -> None:
    matrix = generate_lob_matrix(num_events=500, seed=0)
    expected_rows = NUM_LOB_FEATURES + len(LABEL_HORIZONS)
    assert matrix.shape == (expected_rows, 500)


def test_generator_is_deterministic() -> None:
    first = generate_lob_matrix(num_events=300, seed=11)
    second = generate_lob_matrix(num_events=300, seed=11)
    assert np.array_equal(first, second)


def test_different_seeds_diverge() -> None:
    first = generate_lob_matrix(num_events=300, seed=1)
    second = generate_lob_matrix(num_events=300, seed=2)
    assert not np.array_equal(first, second)


def test_labels_are_one_based_classes() -> None:
    matrix = generate_lob_matrix(num_events=400, seed=5)
    labels = matrix[-len(LABEL_HORIZONS) :, :]
    assert labels.min() >= 1
    assert labels.max() <= 3


def test_rejects_too_few_events() -> None:
    with pytest.raises(ValueError, match="longest horizon"):
        generate_lob_matrix(num_events=max(LABEL_HORIZONS))


def test_rejects_non_positive_features() -> None:
    with pytest.raises(ValueError, match="num_features"):
        generate_lob_matrix(num_events=500, num_features=0)
