"""Sliding-window helpers shared by the dataset.

Rather than materialising every overlapping window up front, which copies
each snapshot ``window`` times, DepthCast keeps the contiguous feature
matrix and slices windows on demand. These small pure functions express
the window arithmetic in one place so the dataset and its tests agree.
"""

from __future__ import annotations


def count_windows(num_events: int, window: int) -> int:
    """Return how many full windows fit in a sequence.

    Args:
        num_events: Total number of snapshots available.
        window: Length of each window.

    Returns:
        The number of windows ``num_events - window + 1``, or ``0`` when
        the sequence is shorter than a single window.

    Raises:
        ValueError: If ``window`` is not positive.
    """
    if window <= 0:
        raise ValueError(f"window must be positive; got {window}.")
    if num_events < window:
        return 0
    return num_events - window + 1


def label_index_for_window(window_index: int, window: int) -> int:
    """Map a window to the snapshot whose label supervises it.

    A window covers rows ``[window_index, window_index + window)`` and is
    supervised by the label aligned to its final snapshot.

    Args:
        window_index: Zero-based index of the window.
        window: Length of each window.

    Returns:
        The row index of the supervising label.

    Raises:
        ValueError: If ``window_index`` is negative or ``window`` is not
            positive.
    """
    if window <= 0:
        raise ValueError(f"window must be positive; got {window}.")
    if window_index < 0:
        raise ValueError(
            f"window_index must be non-negative; got {window_index}."
        )
    return window_index + window - 1
