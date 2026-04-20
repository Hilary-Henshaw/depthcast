"""Synthetic FI-2010-shaped data for tests, examples, and smoke runs.

The real FI-2010 archive is several gigabytes, which is impractical for
unit tests and quick experiments. This generator produces a matrix with
the same layout (feature rows followed by one label row per horizon) by
simulating a slowly drifting mid-price and reading direction labels off
that drift. The output is deterministic given a seed.
"""

from __future__ import annotations

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


def generate_lob_matrix(
    num_events: int,
    num_features: int = NUM_LOB_FEATURES,
    seed: int = 0,
    move_threshold: float = 0.002,
) -> NDArray[np.float64]:
    """Build a deterministic FI-2010-shaped matrix.

    Args:
        num_events: Number of order-book events (columns) to simulate.
        num_features: Number of feature rows to emit.
        seed: Seed for the random generator; identical seeds give
            identical matrices.
        move_threshold: Relative mid-price change above which a horizon
            is labelled "up" or "down" rather than "stationary".

    Returns:
        A ``(num_features + len(LABEL_HORIZONS), num_events)`` matrix of
        ``float64`` values, ready for :func:`split_features_labels`.

    Raises:
        ValueError: If ``num_events`` is too small to cover the longest
            horizon, or if ``num_features`` is not positive.
    """
    if num_features <= 0:
        raise ValueError(f"num_features must be positive; got {num_features}.")
    longest_horizon = max(LABEL_HORIZONS)
    if num_events <= longest_horizon:
        raise ValueError(
            f"num_events must exceed the longest horizon "
            f"({longest_horizon}); got {num_events}."
        )

    rng = np.random.default_rng(seed)
    mid_price = _simulate_mid_price(rng, num_events)
    features = _simulate_features(rng, mid_price, num_features)
    labels = _derive_labels(mid_price, move_threshold)

    matrix = np.vstack([features, labels])
    logger.debug("Generated synthetic matrix %s (seed=%d)", matrix.shape, seed)
    return matrix


def _simulate_mid_price(
    rng: np.random.Generator, num_events: int
) -> NDArray[np.float64]:
    """Produce a positive, slowly drifting mid-price path."""
    steps = rng.normal(loc=0.0, scale=0.01, size=num_events)
    path = np.cumsum(steps)
    prices: NDArray[np.float64] = 100.0 + path - path.min()
    return prices


def _simulate_features(
    rng: np.random.Generator,
    mid_price: NDArray[np.float64],
    num_features: int,
) -> NDArray[np.float64]:
    """Spread the mid-price into noisy depth-like feature rows."""
    offsets = np.linspace(-0.05, 0.05, num_features).reshape(-1, 1)
    base = mid_price.reshape(1, -1) * (1.0 + offsets)
    noise = rng.normal(scale=0.01, size=base.shape)
    return base + noise


def _derive_labels(
    mid_price: NDArray[np.float64], move_threshold: float
) -> NDArray[np.float64]:
    """Read one-based direction labels off the future mid-price.

    The class order matches :data:`MOVEMENT_CLASSES`: index 0 (up),
    1 (stationary), 2 (down), shifted by :data:`RAW_LABEL_OFFSET` to
    mirror the one-based FI-2010 convention.
    """
    num_events = mid_price.shape[0]
    rows = np.empty((_NUM_LABEL_ROWS, num_events), dtype=np.float64)
    for row, horizon in enumerate(LABEL_HORIZONS):
        future = np.empty(num_events, dtype=np.float64)
        future[:-horizon] = mid_price[horizon:]
        future[-horizon:] = mid_price[-1]
        change = (future - mid_price) / mid_price
        classes = np.full(num_events, 1, dtype=np.int64)
        classes[change > move_threshold] = 0
        classes[change < -move_threshold] = 2
        rows[row] = classes + RAW_LABEL_OFFSET
    return rows
