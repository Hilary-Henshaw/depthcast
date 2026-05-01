"""Classification metrics for order-book direction forecasts.

The reference notebook prints accuracy and a scikit-learn report inline.
DepthCast returns those numbers as a typed object so callers can assert
on them, serialise them, or render them however they like, while still
exposing the familiar human-readable report string.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from depthcast.constants import MOVEMENT_CLASSES
from depthcast.logging_utils import get_logger
from depthcast.models.network import DepthCastNet

logger = get_logger(__name__)

_Batch = tuple[torch.Tensor, torch.Tensor]


class ClassificationMetrics(BaseModel):
    """A typed summary of classifier performance on a dataset."""

    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion: list[list[int]]
    report: str


@torch.no_grad()
def collect_predictions(
    model: DepthCastNet,
    loader: DataLoader[_Batch],
    device: torch.device,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Run a model over a loader and gather targets and predictions.

    Args:
        model: The network to evaluate.
        loader: Loader yielding ``(features, label)`` batches.
        device: Device to run inference on.

    Returns:
        A pair ``(targets, predictions)`` of 1-D integer arrays.

    Raises:
        ValueError: If the loader yields no samples.
    """
    model.eval()
    model.to(device)
    targets: list[NDArray[np.int64]] = []
    predictions: list[NDArray[np.int64]] = []

    for inputs, batch_targets in loader:
        inputs = inputs.to(device, dtype=torch.float32)
        logits = model(inputs)
        batch_predictions = torch.argmax(logits, dim=1)
        targets.append(batch_targets.cpu().numpy().astype(np.int64))
        predictions.append(batch_predictions.cpu().numpy().astype(np.int64))

    if not targets:
        raise ValueError("Loader yielded no samples to evaluate.")
    return np.concatenate(targets), np.concatenate(predictions)


def evaluate_classifier(
    model: DepthCastNet,
    loader: DataLoader[_Batch],
    device: torch.device,
    class_names: tuple[str, ...] = MOVEMENT_CLASSES,
) -> ClassificationMetrics:
    """Compute a full metric summary for a model on a dataset.

    Args:
        model: The network to evaluate.
        loader: Loader yielding evaluation batches.
        device: Device to run inference on.
        class_names: Human-readable names for each class index.

    Returns:
        A populated :class:`ClassificationMetrics`.
    """
    targets, predictions = collect_predictions(model, loader, device)
    labels = list(range(len(class_names)))

    accuracy = float(accuracy_score(targets, predictions))
    macro_f1 = float(
        f1_score(
            targets,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    per_class = f1_score(
        targets,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class_f1 = {
        name: float(value)
        for name, value in zip(class_names, per_class, strict=True)
    }
    confusion = confusion_matrix(targets, predictions, labels=labels).tolist()
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )

    logger.info(
        "Evaluation complete: accuracy=%.4f macro_f1=%.4f",
        accuracy,
        macro_f1,
    )
    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        confusion=confusion,
        report=str(report),
    )
