"""Tests for the lazy windowed dataset and loader factory."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from depthcast.data.dataset import LobWindowDataset, build_dataloader


def _arrays(
    num_events: int = 60, num_features: int = 40
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    features = np.arange(num_events * num_features, dtype=np.float32).reshape(
        num_events, num_features
    )
    labels = np.arange(num_events, dtype=np.int64) % 3
    return features, labels


def test_length_matches_window_count() -> None:
    features, labels = _arrays(num_events=60)
    dataset = LobWindowDataset(features, labels, window=50)
    assert len(dataset) == 11
    assert dataset.num_features == 40


def test_getitem_returns_channel_first_window() -> None:
    features, labels = _arrays(num_events=60)
    dataset = LobWindowDataset(features, labels, window=50)
    window, label = dataset[0]
    assert window.shape == (1, 50, 40)
    assert window.dtype == torch.float32
    assert label.item() == labels[49]


def test_negative_index_wraps() -> None:
    features, labels = _arrays(num_events=60)
    dataset = LobWindowDataset(features, labels, window=50)
    last_window, last_label = dataset[-1]
    direct_window, direct_label = dataset[len(dataset) - 1]
    assert torch.equal(last_window, direct_window)
    assert last_label.item() == direct_label.item()


def test_out_of_range_index_raises() -> None:
    features, labels = _arrays(num_events=60)
    dataset = LobWindowDataset(features, labels, window=50)
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_mismatched_lengths_raise() -> None:
    features, labels = _arrays(num_events=60)
    with pytest.raises(ValueError, match="share the event axis"):
        LobWindowDataset(features, labels[:-1], window=50)


def test_non_two_dimensional_features_raise() -> None:
    _, labels = _arrays(num_events=60)
    with pytest.raises(ValueError, match="2-D"):
        LobWindowDataset(np.ones(60, dtype=np.float32), labels, window=50)


def test_non_one_dimensional_labels_raise() -> None:
    features, _ = _arrays(num_events=60)
    bad_labels = np.zeros((60, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="1-D"):
        LobWindowDataset(features, bad_labels, window=50)


def test_window_longer_than_session_raises() -> None:
    features, labels = _arrays(num_events=20)
    with pytest.raises(ValueError, match="shorter than"):
        LobWindowDataset(features, labels, window=50)


def test_build_dataloader_batches_windows() -> None:
    features, labels = _arrays(num_events=120)
    dataset = LobWindowDataset(features, labels, window=50)
    loader = build_dataloader(dataset, batch_size=16, shuffle=False)
    inputs, targets = next(iter(loader))
    assert inputs.shape == (16, 1, 50, 40)
    assert targets.shape == (16,)


def test_build_dataloader_rejects_non_positive_batch() -> None:
    features, labels = _arrays(num_events=60)
    dataset = LobWindowDataset(features, labels, window=50)
    with pytest.raises(ValueError, match="batch_size"):
        build_dataloader(dataset, batch_size=0, shuffle=False)
