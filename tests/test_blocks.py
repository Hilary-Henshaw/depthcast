"""Tests for the convolutional building blocks."""

from __future__ import annotations

import pytest
import torch

from depthcast.models.blocks import (
    TIME_STEPS_CONSUMED,
    DepthCompressionStack,
    MultiScaleFusion,
    _activation,
)


def test_compression_collapses_feature_axis() -> None:
    stack = DepthCompressionStack(channels=8, num_features=40, slope=0.01)
    window = 50
    output = stack(torch.randn(2, 1, window, 40))
    assert output.shape == (2, 8, window - TIME_STEPS_CONSUMED, 1)


def test_compression_rejects_indivisible_features() -> None:
    with pytest.raises(ValueError, match="divisible"):
        DepthCompressionStack(channels=8, num_features=41, slope=0.01)


def test_fusion_concatenates_three_branches() -> None:
    fusion = MultiScaleFusion(in_channels=8, branch_channels=16, slope=0.01)
    output = fusion(torch.randn(2, 8, 32, 1))
    assert output.shape == (2, 48, 32, 1)


def test_activation_factory_supports_known_kinds() -> None:
    assert isinstance(_activation("tanh", 0.01), torch.nn.Tanh)
    assert isinstance(_activation("leaky_relu", 0.01), torch.nn.LeakyReLU)


def test_activation_factory_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported activation"):
        _activation("swish", 0.01)
