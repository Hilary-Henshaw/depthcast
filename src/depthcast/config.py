"""Typed, validated configuration for DepthCast experiments.

Every tunable parameter lives in a Pydantic model so that invalid runs
fail fast with a precise message instead of surfacing as an obscure shape
error deep inside the network. Configurations round-trip through YAML,
which keeps experiment definitions reviewable and reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from depthcast.constants import (
    FEATURE_COMPRESSION_FACTOR,
    LABEL_HORIZONS,
    NUM_CLASSES,
    NUM_LOB_FEATURES,
)

# The convolutional stack removes a fixed number of time steps: each of
# the three stages applies two height-4 convolutions without padding.
_TIME_STEPS_CONSUMED: int = 18

DeviceChoice = Literal["auto", "cpu", "cuda"]


class DataConfig(BaseModel):
    """How raw order-book matrices become supervised windows."""

    model_config = ConfigDict(extra="forbid")

    window: int = Field(
        default=100,
        ge=_TIME_STEPS_CONSUMED + 1,
        description="Number of consecutive snapshots per sample.",
    )
    horizon_index: int = Field(
        default=4,
        ge=0,
        le=len(LABEL_HORIZONS) - 1,
        description="Which FI-2010 horizon label to predict.",
    )
    num_features: int = Field(
        default=NUM_LOB_FEATURES,
        gt=0,
        description="Order-book feature columns per snapshot.",
    )
    batch_size: int = Field(default=64, gt=0)
    num_workers: int = Field(default=0, ge=0)
    shuffle_train: bool = Field(default=True)

    @model_validator(mode="after")
    def _check_feature_divisibility(self) -> DataConfig:
        if self.num_features % FEATURE_COMPRESSION_FACTOR != 0:
            raise ValueError(
                "num_features must be divisible by "
                f"{FEATURE_COMPRESSION_FACTOR}; got {self.num_features}."
            )
        return self

    @property
    def horizon_ticks(self) -> int:
        """The prediction horizon, expressed in order-book events."""
        return LABEL_HORIZONS[self.horizon_index]


class ModelConfig(BaseModel):
    """Architecture hyper-parameters for :class:`DepthCastNet`."""

    model_config = ConfigDict(extra="forbid")

    conv_channels: int = Field(default=32, gt=0)
    fusion_channels: int = Field(default=64, gt=0)
    lstm_hidden: int = Field(default=64, gt=0)
    lstm_layers: int = Field(default=1, gt=0)
    num_classes: int = Field(default=NUM_CLASSES, ge=2)
    leaky_slope: float = Field(default=0.01, ge=0.0, lt=1.0)
    recurrent_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    num_features: int = Field(default=NUM_LOB_FEATURES, gt=0)

    @model_validator(mode="after")
    def _check_feature_divisibility(self) -> ModelConfig:
        if self.num_features % FEATURE_COMPRESSION_FACTOR != 0:
            raise ValueError(
                "num_features must be divisible by "
                f"{FEATURE_COMPRESSION_FACTOR}; got {self.num_features}."
            )
        return self

    @property
    def fusion_width(self) -> int:
        """Channel count after the three fusion branches concatenate."""
        return self.fusion_channels * 3


class TrainConfig(BaseModel):
    """Optimisation, checkpointing, and runtime settings."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=50, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    grad_clip: float | None = Field(default=None, gt=0.0)
    early_stop_patience: int | None = Field(default=10, gt=0)
    seed: int = Field(default=42, ge=0)
    device: DeviceChoice = Field(default="auto")
    checkpoint_dir: Path = Field(default=Path("checkpoints"))
    log_level: str = Field(default="INFO")

    @model_validator(mode="after")
    def _validate_log_level(self) -> TrainConfig:
        import logging

        if not isinstance(logging.getLevelName(self.log_level.upper()), int):
            raise ValueError(f"Unknown log level: {self.log_level!r}")
        return self


class ExperimentConfig(BaseModel):
    """The full configuration for a single training run."""

    model_config = ConfigDict(extra="forbid")

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @model_validator(mode="after")
    def _cross_validate(self) -> ExperimentConfig:
        if self.data.num_features != self.model.num_features:
            raise ValueError(
                "data.num_features and model.num_features must match; "
                f"got {self.data.num_features} and "
                f"{self.model.num_features}."
            )
        if self.data.window <= _TIME_STEPS_CONSUMED:
            raise ValueError(
                "data.window must exceed the "
                f"{_TIME_STEPS_CONSUMED} time steps consumed by the "
                "convolutional stack."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate a configuration from a YAML file.

        Args:
            path: Path to a YAML document with optional ``data``,
                ``model``, and ``train`` mappings.

        Returns:
            A fully validated :class:`ExperimentConfig`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the document is not a mapping.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
        raw: Any = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"Config root must be a mapping, got {type(raw).__name__}."
            )
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        """Serialise this configuration to a YAML file.

        Args:
            path: Destination path; parent directories are created.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        out_path.write_text(yaml.safe_dump(payload, sort_keys=False, indent=2))
