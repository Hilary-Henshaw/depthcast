"""The DepthCast training loop.

:class:`Trainer` owns the optimisation state, runs epochs, tracks the
best validation loss, checkpoints it, and stops early when validation
stops improving. It reports progress through structured log records and
returns a typed history rather than printing or plotting inline.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader

from depthcast.config import ExperimentConfig
from depthcast.logging_utils import get_logger
from depthcast.models.network import DepthCastNet
from depthcast.runtime import resolve_device, seed_everything
from depthcast.training.checkpoint import save_checkpoint

logger = get_logger(__name__)

_Batch = tuple[torch.Tensor, torch.Tensor]


class EpochReport(BaseModel):
    """A single epoch's metrics, returned by :meth:`Trainer.fit`."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    is_best: bool
    duration_seconds: float


class Trainer:
    """Coordinate optimisation, evaluation, and checkpointing."""

    def __init__(
        self,
        model: DepthCastNet,
        config: ExperimentConfig,
        device: torch.device | None = None,
    ) -> None:
        """Bind a model and configuration to an optimiser.

        Args:
            model: The network to train.
            config: The full experiment configuration.
            device: Device to train on; resolved from the config when
                omitted.
        """
        self.config = config
        self.device = device or resolve_device(config.train.device)
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self._best_val_loss = math.inf
        self._best_path: Path | None = None
        logger.debug("Trainer ready on device %s", self.device)

    @classmethod
    def build(cls, config: ExperimentConfig) -> Trainer:
        """Seed generators, construct the model, and return a trainer.

        Seeding before the model is created makes weight initialisation
        reproducible for a given configuration.

        Args:
            config: The full experiment configuration.

        Returns:
            A ready-to-fit trainer.
        """
        seed_everything(config.train.seed)
        model = DepthCastNet(config.model)
        return cls(model, config)

    @property
    def best_checkpoint_path(self) -> Path | None:
        """Path of the best checkpoint written so far, if any."""
        return self._best_path

    def fit(
        self,
        train_loader: DataLoader[_Batch],
        val_loader: DataLoader[_Batch],
    ) -> list[EpochReport]:
        """Train for the configured number of epochs with early stop.

        Args:
            train_loader: Loader yielding training batches.
            val_loader: Loader yielding validation batches.

        Returns:
            One :class:`EpochReport` per epoch actually run.
        """
        checkpoint_dir = self.config.train.checkpoint_dir
        patience = self.config.train.early_stop_patience
        history: list[EpochReport] = []
        epochs_without_gain = 0

        for epoch in range(1, self.config.train.epochs + 1):
            started = time.perf_counter()
            train_loss = self._train_one_epoch(train_loader)
            val_loss, val_accuracy = self._evaluate(val_loader)
            is_best = val_loss < self._best_val_loss

            if is_best:
                self._best_val_loss = val_loss
                epochs_without_gain = 0
                self._best_path = save_checkpoint(
                    checkpoint_dir / "best.pt",
                    self.model,
                    self.config,
                    epoch,
                    val_loss,
                )
            else:
                epochs_without_gain += 1

            report = EpochReport(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                is_best=is_best,
                duration_seconds=time.perf_counter() - started,
            )
            history.append(report)
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "val_acc=%.4f | best=%s",
                epoch,
                self.config.train.epochs,
                train_loss,
                val_loss,
                val_accuracy,
                is_best,
            )

            if patience is not None and epochs_without_gain >= patience:
                logger.info(
                    "Stopping early after %d epochs without improvement.",
                    patience,
                )
                break

        return history

    def _train_one_epoch(self, loader: DataLoader[_Batch]) -> float:
        """Run one optimisation pass and return the mean loss."""
        self.model.train()
        running_loss = 0.0
        seen = 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            if self.config.train.grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.train.grad_clip,
                )
            self.optimizer.step()

            batch_size = targets.shape[0]
            running_loss += loss.item() * batch_size
            seen += batch_size
        return running_loss / max(seen, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader[_Batch]) -> tuple[float, float]:
        """Return the mean loss and accuracy over a loader."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        seen = 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.long)

            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            predictions = torch.argmax(logits, dim=1)

            batch_size = targets.shape[0]
            running_loss += loss.item() * batch_size
            correct += int((predictions == targets).sum().item())
            seen += batch_size

        mean_loss = running_loss / max(seen, 1)
        accuracy = correct / max(seen, 1)
        return mean_loss, accuracy
