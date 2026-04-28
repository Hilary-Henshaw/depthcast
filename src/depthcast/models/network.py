"""The DepthCast forecasting network.

:class:`DepthCastNet` wires the convolutional compression stack and the
multi-scale fusion module into a recurrent classifier. Unlike the
reference implementation it returns raw class logits rather than softmax
probabilities, so it composes correctly with
:class:`torch.nn.CrossEntropyLoss` without applying the softmax twice.
"""

from __future__ import annotations

import torch
from torch import nn

from depthcast.config import ModelConfig
from depthcast.logging_utils import get_logger
from depthcast.models.blocks import (
    TIME_STEPS_CONSUMED,
    DepthCompressionStack,
    MultiScaleFusion,
)

logger = get_logger(__name__)


class DepthCastNet(nn.Module):
    """Convolutional-recurrent order-book direction classifier."""

    def __init__(self, config: ModelConfig) -> None:
        """Assemble the network from a validated configuration.

        Args:
            config: Architecture hyper-parameters.
        """
        super().__init__()
        self.config = config

        self.compression = DepthCompressionStack(
            channels=config.conv_channels,
            num_features=config.num_features,
            slope=config.leaky_slope,
        )
        self.fusion = MultiScaleFusion(
            in_channels=config.conv_channels,
            branch_channels=config.fusion_channels,
            slope=config.leaky_slope,
        )
        lstm_dropout = (
            config.recurrent_dropout if config.lstm_layers > 1 else 0.0
        )
        self.recurrent = nn.LSTM(
            input_size=config.fusion_width,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Linear(config.lstm_hidden, config.num_classes)
        logger.debug(
            "Initialised DepthCastNet with fusion width %d",
            config.fusion_width,
        )

    @classmethod
    def from_config(cls, config: ModelConfig) -> DepthCastNet:
        """Alias constructor for symmetry with the rest of the API."""
        return cls(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map order-book windows to class logits.

        Args:
            inputs: Tensor shaped ``(batch, 1, window, num_features)``.

        Returns:
            Logits shaped ``(batch, num_classes)``.

        Raises:
            ValueError: If ``inputs`` does not have four dimensions or
                the window is too short for the compression stack.
        """
        self._check_input(inputs)
        spatial = self.compression(inputs)
        fused = self.fusion(spatial)

        # (batch, channels, time, 1) -> (batch, time, channels)
        sequence = fused.permute(0, 2, 1, 3).squeeze(-1)
        recurrent_out, _ = self.recurrent(sequence)
        last_step = recurrent_out[:, -1, :]
        logits: torch.Tensor = self.head(last_step)
        return logits

    @torch.no_grad()
    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class probabilities for a batch of windows.

        Args:
            inputs: Tensor shaped ``(batch, 1, window, num_features)``.

        Returns:
            Probabilities shaped ``(batch, num_classes)`` summing to one.
        """
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(inputs)
            return torch.softmax(logits, dim=1)
        finally:
            self.train(was_training)

    def _check_input(self, inputs: torch.Tensor) -> None:
        """Validate the rank and window length of an input batch."""
        if inputs.dim() != 4:
            raise ValueError(
                "Expected a 4-D tensor "
                "(batch, 1, window, features); "
                f"got {inputs.dim()} dimensions."
            )
        window = inputs.shape[2]
        if window <= TIME_STEPS_CONSUMED:
            raise ValueError(
                f"window must exceed {TIME_STEPS_CONSUMED} time steps "
                f"consumed by the convolutions; got {window}."
            )
        features = inputs.shape[3]
        if features != self.config.num_features:
            raise ValueError(
                f"Expected {self.config.num_features} features; "
                f"got {features}."
            )
