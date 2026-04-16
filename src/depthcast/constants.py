"""Domain constants shared across the DepthCast pipeline.

These values describe the FI-2010 limit-order-book benchmark and the
fixed shape of the forecasting problem. They are intentionally kept in a
single module so that the data, model, and configuration layers agree on
the same definitions instead of re-deriving them independently.
"""

from __future__ import annotations

from typing import Final

# The FI-2010 feature matrix stores ten order-book levels, each described
# by an ask price, ask volume, bid price, and bid volume: 10 * 4 = 40.
NUM_LOB_FEATURES: Final[int] = 40

# Each price/volume pair plus each level is folded away by a stride-2
# convolution, so the raw feature axis must be divisible by this factor.
FEATURE_COMPRESSION_FACTOR: Final[int] = 4

# Prediction horizons (in events) shipped with the FI-2010 labels. The
# dataset stores one label row per horizon, in this order.
LABEL_HORIZONS: Final[tuple[int, ...]] = (10, 20, 30, 50, 100)

# Three mutually exclusive movement classes. The index order matches the
# zero-based labels produced after decoding the FI-2010 annotations.
MOVEMENT_CLASSES: Final[tuple[str, ...]] = ("up", "stationary", "down")

NUM_CLASSES: Final[int] = len(MOVEMENT_CLASSES)

# Raw FI-2010 annotations are one-based; subtracting this yields the
# zero-based class index used by the loss function.
RAW_LABEL_OFFSET: Final[int] = 1
