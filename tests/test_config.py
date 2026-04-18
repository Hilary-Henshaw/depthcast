"""Tests for the typed configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from depthcast.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)


def test_default_experiment_config_is_valid() -> None:
    config = ExperimentConfig()
    assert config.data.window == 100
    assert config.model.num_classes == 3
    assert config.train.epochs == 50


def test_horizon_ticks_maps_index_to_event_count() -> None:
    config = DataConfig(horizon_index=4)
    assert config.horizon_ticks == 100
    assert DataConfig(horizon_index=0).horizon_ticks == 10


def test_fusion_width_is_three_branches() -> None:
    config = ModelConfig(fusion_channels=64)
    assert config.fusion_width == 192


def test_data_config_rejects_indivisible_features() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        DataConfig(num_features=41)


def test_model_config_rejects_indivisible_features() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        ModelConfig(num_features=41)


def test_window_below_minimum_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DataConfig(window=5)


def test_cross_validation_requires_matching_features() -> None:
    with pytest.raises(ValidationError, match="must match"):
        ExperimentConfig(
            data=DataConfig(num_features=40),
            model=ModelConfig(num_features=80),
        )


def test_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError, match="log level"):
        TrainConfig(log_level="LOUD")


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        DataConfig(unexpected=1)  # type: ignore[call-arg]


def test_yaml_round_trip_preserves_values(tmp_path: Path) -> None:
    original = ExperimentConfig(
        data=DataConfig(window=64, horizon_index=2),
        train=TrainConfig(epochs=7, learning_rate=5e-4),
    )
    path = tmp_path / "nested" / "config.yaml"
    original.to_yaml(path)
    assert path.is_file()

    restored = ExperimentConfig.from_yaml(path)
    assert restored == original


def test_from_yaml_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.from_yaml(tmp_path / "absent.yaml")


def test_from_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        ExperimentConfig.from_yaml(path)


def test_from_yaml_accepts_empty_document(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    config = ExperimentConfig.from_yaml(path)
    assert config == ExperimentConfig()
