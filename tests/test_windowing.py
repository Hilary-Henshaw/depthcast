"""Tests for the sliding-window arithmetic."""

from __future__ import annotations

import pytest

from depthcast.data.windowing import count_windows, label_index_for_window


def test_count_windows_basic() -> None:
    assert count_windows(10, 3) == 8


def test_count_windows_exact_fit() -> None:
    assert count_windows(5, 5) == 1


def test_count_windows_too_short_returns_zero() -> None:
    assert count_windows(2, 5) == 0


def test_count_windows_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="positive"):
        count_windows(10, 0)


def test_label_index_points_at_window_end() -> None:
    assert label_index_for_window(0, 100) == 99
    assert label_index_for_window(5, 10) == 14


def test_label_index_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        label_index_for_window(-1, 10)


def test_label_index_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="positive"):
        label_index_for_window(0, 0)
