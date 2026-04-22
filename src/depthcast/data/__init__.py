"""Data ingestion and windowing for DepthCast."""

from __future__ import annotations

from depthcast.data.dataset import LobWindowDataset, build_dataloader
from depthcast.data.fi2010 import load_lob_matrix, split_features_labels
from depthcast.data.synthetic import generate_lob_matrix
from depthcast.data.windowing import count_windows

__all__ = [
    "LobWindowDataset",
    "build_dataloader",
    "count_windows",
    "generate_lob_matrix",
    "load_lob_matrix",
    "split_features_labels",
]
