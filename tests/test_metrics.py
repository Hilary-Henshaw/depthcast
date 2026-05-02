"""Tests for the evaluation metrics."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from depthcast.config import ExperimentConfig
from depthcast.constants import MOVEMENT_CLASSES
from depthcast.evaluation.metrics import (
    collect_predictions,
    evaluate_classifier,
)
from depthcast.models.network import DepthCastNet
from depthcast.pipeline import make_loader, matrix_to_dataset
from depthcast.runtime import resolve_device


def test_evaluate_classifier_reports_all_classes(
    fast_config: ExperimentConfig,
    synthetic_matrix: NDArray[np.float64],
) -> None:
    model = DepthCastNet(fast_config.model)
    loader = make_loader(
        matrix_to_dataset(synthetic_matrix, fast_config.data),
        fast_config.data,
        shuffle=False,
    )
    metrics = evaluate_classifier(model, loader, resolve_device("cpu"))

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert set(metrics.per_class_f1) == set(MOVEMENT_CLASSES)
    assert len(metrics.confusion) == len(MOVEMENT_CLASSES)
    assert all(len(row) == len(MOVEMENT_CLASSES) for row in metrics.confusion)
    assert metrics.report.strip()


def test_collect_predictions_rejects_empty_loader(
    fast_config: ExperimentConfig,
) -> None:
    model = DepthCastNet(fast_config.model)
    empty = DataLoader(
        TensorDataset(
            torch.empty(0, 1, fast_config.data.window, 40),
            torch.empty(0, dtype=torch.long),
        )
    )
    typed_empty = cast("DataLoader[tuple[torch.Tensor, torch.Tensor]]", empty)
    with pytest.raises(ValueError, match="no samples"):
        collect_predictions(model, typed_empty, resolve_device("cpu"))


def test_collect_predictions_shapes_match(
    fast_config: ExperimentConfig,
    synthetic_matrix: NDArray[np.float64],
) -> None:
    model = DepthCastNet(fast_config.model)
    loader = make_loader(
        matrix_to_dataset(synthetic_matrix, fast_config.data),
        fast_config.data,
        shuffle=False,
    )
    targets, predictions = collect_predictions(
        model, loader, resolve_device("cpu")
    )
    assert targets.shape == predictions.shape
    assert targets.dtype == np.int64
