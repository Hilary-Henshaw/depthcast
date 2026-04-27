"""Tests for the DepthCast network."""

from __future__ import annotations

import pytest
import torch

from depthcast.config import ModelConfig
from depthcast.models.network import DepthCastNet


def _model(**overrides: object) -> DepthCastNet:
    config = ModelConfig(
        conv_channels=8, fusion_channels=8, lstm_hidden=8, **overrides
    )
    return DepthCastNet(config)


def test_forward_returns_class_logits() -> None:
    model = _model()
    logits = model(torch.randn(4, 1, 50, 40))
    assert logits.shape == (4, 3)


def test_from_config_builds_equivalent_model() -> None:
    config = ModelConfig(conv_channels=8, fusion_channels=8)
    model = DepthCastNet.from_config(config)
    assert isinstance(model, DepthCastNet)


def test_predict_proba_rows_sum_to_one() -> None:
    model = _model()
    proba = model.predict_proba(torch.randn(3, 1, 50, 40))
    assert proba.shape == (3, 3)
    assert torch.allclose(proba.sum(dim=1), torch.ones(3), atol=1e-5)
    assert torch.all(proba >= 0)


def test_predict_proba_restores_training_mode() -> None:
    model = _model()
    model.train()
    model.predict_proba(torch.randn(2, 1, 50, 40))
    assert model.training is True
    model.eval()
    model.predict_proba(torch.randn(2, 1, 50, 40))
    assert model.training is False


def test_forward_rejects_wrong_rank() -> None:
    model = _model()
    with pytest.raises(ValueError, match="4-D"):
        model(torch.randn(4, 50, 40))


def test_forward_rejects_short_window() -> None:
    model = _model()
    with pytest.raises(ValueError, match="window must exceed"):
        model(torch.randn(4, 1, 10, 40))


def test_forward_rejects_feature_mismatch() -> None:
    model = _model()
    with pytest.raises(ValueError, match="features"):
        model(torch.randn(4, 1, 50, 80))


def test_stacked_lstm_with_dropout_builds() -> None:
    model = _model(lstm_layers=2, recurrent_dropout=0.3)
    logits = model(torch.randn(2, 1, 50, 40))
    assert logits.shape == (2, 3)
