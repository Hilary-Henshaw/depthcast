"""Convolutional building blocks for the DepthCast network.

Two modules carry the spatial reasoning. :class:`DepthCompressionStack`
folds the raw order-book feature axis down to a single column while
preserving the time axis, and :class:`MultiScaleFusion` reads that
representation at several temporal receptive fields in parallel.
"""

from __future__ import annotations

import torch
from torch import nn

from depthcast.constants import FEATURE_COMPRESSION_FACTOR

# Each compression stage applies two height-4 convolutions without
# padding, so every stage shortens the time axis by this many steps.
TIME_STEPS_PER_STAGE: int = 6
_NUM_STAGES: int = 3
TIME_STEPS_CONSUMED: int = TIME_STEPS_PER_STAGE * _NUM_STAGES


def _activation(kind: str, slope: float) -> nn.Module:
    """Build an activation module by name.

    Args:
        kind: Either ``"leaky_relu"`` or ``"tanh"``.
        slope: Negative slope used when ``kind`` is ``"leaky_relu"``.

    Returns:
        The corresponding activation module.

    Raises:
        ValueError: If ``kind`` is not a supported activation.
    """
    if kind == "leaky_relu":
        return nn.LeakyReLU(negative_slope=slope)
    if kind == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {kind!r}")


def _conv_unit(
    in_channels: int,
    out_channels: int,
    kernel: tuple[int, int],
    activation: str,
    slope: float,
    stride: tuple[int, int] = (1, 1),
) -> list[nn.Module]:
    """Return a convolution followed by activation and batch norm."""
    return [
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride),
        _activation(activation, slope),
        nn.BatchNorm2d(out_channels),
    ]


class DepthCompressionStack(nn.Module):
    """Collapse the feature axis to width one over three stages.

    The first convolution of each stage halves the feature axis (or, in
    the final stage, removes it entirely), while paired height-4
    convolutions summarise local temporal structure. The middle stage
    uses ``tanh`` activations to bound the intermediate representation,
    matching the published DeepLOB topology.
    """

    def __init__(
        self,
        channels: int,
        num_features: int,
        slope: float,
    ) -> None:
        """Build the three-stage compression tower.

        Args:
            channels: Convolution channel width used throughout.
            num_features: Width of the raw feature axis; must be
                divisible by :data:`FEATURE_COMPRESSION_FACTOR`.
            slope: Negative slope for the leaky activations.

        Raises:
            ValueError: If ``num_features`` is not divisible by the
                compression factor.
        """
        super().__init__()
        if num_features % FEATURE_COMPRESSION_FACTOR != 0:
            raise ValueError(
                "num_features must be divisible by "
                f"{FEATURE_COMPRESSION_FACTOR}; got {num_features}."
            )
        level_kernel = num_features // FEATURE_COMPRESSION_FACTOR

        self.stage_pair = nn.Sequential(
            *_conv_unit(1, channels, (1, 2), "leaky_relu", slope, (1, 2)),
            *_conv_unit(channels, channels, (4, 1), "leaky_relu", slope),
            *_conv_unit(channels, channels, (4, 1), "leaky_relu", slope),
        )
        self.stage_level = nn.Sequential(
            *_conv_unit(channels, channels, (1, 2), "tanh", slope, (1, 2)),
            *_conv_unit(channels, channels, (4, 1), "tanh", slope),
            *_conv_unit(channels, channels, (4, 1), "tanh", slope),
        )
        self.stage_collapse = nn.Sequential(
            *_conv_unit(
                channels, channels, (1, level_kernel), "leaky_relu", slope
            ),
            *_conv_unit(channels, channels, (4, 1), "leaky_relu", slope),
            *_conv_unit(channels, channels, (4, 1), "leaky_relu", slope),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compress ``(batch, 1, time, features)`` to width one.

        Args:
            inputs: Tensor shaped ``(batch, 1, time, num_features)``.

        Returns:
            Tensor shaped ``(batch, channels, time - 18, 1)``.
        """
        compressed = self.stage_pair(inputs)
        compressed = self.stage_level(compressed)
        collapsed: torch.Tensor = self.stage_collapse(compressed)
        return collapsed


class MultiScaleFusion(nn.Module):
    """Read the compressed sequence at three temporal scales.

    Three parallel branches (a 3-step branch, a 5-step branch, and a
    pooled branch) view the same input and are concatenated on the
    channel axis, giving the recurrent layer a multi-resolution summary
    of each time step.
    """

    def __init__(
        self, in_channels: int, branch_channels: int, slope: float
    ) -> None:
        """Build the three fusion branches.

        Args:
            in_channels: Channels arriving from the compression stack.
            branch_channels: Output channels for each branch.
            slope: Negative slope for the leaky activations.
        """
        super().__init__()
        self.branch_short = nn.Sequential(
            *_same_conv(in_channels, branch_channels, (1, 1), slope),
            *_same_conv(branch_channels, branch_channels, (3, 1), slope),
        )
        self.branch_long = nn.Sequential(
            *_same_conv(in_channels, branch_channels, (1, 1), slope),
            *_same_conv(branch_channels, branch_channels, (5, 1), slope),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            *_same_conv(in_channels, branch_channels, (1, 1), slope),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply every branch and concatenate on the channel axis.

        Args:
            inputs: Tensor shaped ``(batch, in_channels, time, 1)``.

        Returns:
            Tensor shaped ``(batch, 3 * branch_channels, time, 1)``.
        """
        return torch.cat(
            (
                self.branch_short(inputs),
                self.branch_long(inputs),
                self.branch_pool(inputs),
            ),
            dim=1,
        )


def _same_conv(
    in_channels: int,
    out_channels: int,
    kernel: tuple[int, int],
    slope: float,
) -> list[nn.Module]:
    """Return a padding-preserving conv, activation, and batch norm."""
    return [
        nn.Conv2d(in_channels, out_channels, kernel, padding="same"),
        nn.LeakyReLU(negative_slope=slope),
        nn.BatchNorm2d(out_channels),
    ]
