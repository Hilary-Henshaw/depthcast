"""End-to-end tests for the command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from depthcast.cli import app
from depthcast.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)

runner = CliRunner()


def _tiny_config_file(tmp_path: Path) -> Path:
    config = ExperimentConfig(
        data=DataConfig(window=50, batch_size=32),
        model=ModelConfig(conv_channels=4, fusion_channels=4, lstm_hidden=8),
        train=TrainConfig(
            epochs=1,
            early_stop_patience=None,
            device="cpu",
            checkpoint_dir=tmp_path / "ckpt",
        ),
    )
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    return path


def test_init_config_writes_file(tmp_path: Path) -> None:
    target = tmp_path / "out.yaml"
    result = runner.invoke(app, ["init-config", str(target)])
    assert result.exit_code == 0
    assert target.is_file()
    ExperimentConfig.from_yaml(target)


def test_synth_writes_session(tmp_path: Path) -> None:
    target = tmp_path / "session.txt"
    result = runner.invoke(
        app,
        ["synth", "--out", str(target), "--events", "400"],
    )
    assert result.exit_code == 0
    assert target.is_file()


def test_summary_reports_parameters(tmp_path: Path) -> None:
    config_path = _tiny_config_file(tmp_path)
    result = runner.invoke(app, ["summary", "--config", str(config_path)])
    assert result.exit_code == 0
    assert "Total parameters" in result.stdout


@pytest.mark.integration
def test_train_then_evaluate_flow(tmp_path: Path) -> None:
    config_path = _tiny_config_file(tmp_path)

    train_result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(config_path),
            "--synthetic",
            "--synthetic-events",
            "600",
        ],
    )
    assert train_result.exit_code == 0, train_result.stdout
    checkpoint = tmp_path / "ckpt" / "best.pt"
    assert checkpoint.is_file()

    eval_result = runner.invoke(
        app,
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--synthetic",
            "--synthetic-events",
            "600",
        ],
    )
    assert eval_result.exit_code == 0, eval_result.stdout
    assert "Accuracy" in eval_result.stdout


def test_train_requires_a_data_source(tmp_path: Path) -> None:
    config_path = _tiny_config_file(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code != 0
