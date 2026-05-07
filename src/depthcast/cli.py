"""Command-line interface for DepthCast.

The CLI exposes the whole workflow: scaffold a configuration, synthesise
practice data, inspect the architecture, train a model, and evaluate a
checkpoint. Progress is logged through the structured logger while final
results are written to standard output for the user.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from numpy.typing import NDArray

from depthcast.config import ExperimentConfig
from depthcast.data.fi2010 import load_lob_matrix
from depthcast.data.synthetic import generate_lob_matrix
from depthcast.evaluation.metrics import evaluate_classifier
from depthcast.logging_utils import configure_logging, get_logger
from depthcast.models.network import DepthCastNet
from depthcast.pipeline import make_loader, matrix_to_dataset, split_session
from depthcast.runtime import resolve_device
from depthcast.training.checkpoint import load_checkpoint
from depthcast.training.trainer import Trainer

logger = get_logger(__name__)

app = typer.Typer(
    name="depthcast",
    help="Forecast limit-order-book price direction with DepthCast.",
    add_completion=True,
    no_args_is_help=True,
)


def _load_config(path: Path | None) -> ExperimentConfig:
    """Load a config from ``path`` or return the validated default."""
    if path is None:
        logger.info("No config supplied; using built-in defaults.")
        return ExperimentConfig()
    return ExperimentConfig.from_yaml(path)


@app.command("init-config")
def init_config(
    path: Annotated[
        Path, typer.Argument(help="Where to write the YAML config.")
    ] = Path("depthcast.yaml"),
) -> None:
    """Write a default experiment configuration to disk."""
    configure_logging()
    ExperimentConfig().to_yaml(path)
    typer.echo(f"Wrote default configuration to {path}")


@app.command("synth")
def synth(
    out: Annotated[Path, typer.Option(help="Destination text file.")] = Path(
        "synthetic_session.txt"
    ),
    events: Annotated[
        int, typer.Option(help="Number of order-book events.")
    ] = 20_000,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
) -> None:
    """Generate a synthetic FI-2010-shaped session for experiments."""
    configure_logging()
    matrix = generate_lob_matrix(num_events=events, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, matrix)
    typer.echo(f"Wrote synthetic session {matrix.shape} to {out}")


@app.command("summary")
def summary(
    config: Annotated[
        Path | None, typer.Option(help="Config YAML; default if unset.")
    ] = None,
) -> None:
    """Print the network architecture and parameter count."""
    configure_logging()
    experiment = _load_config(config)
    model = DepthCastNet(experiment.model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    typer.echo(model)
    typer.echo(f"Total parameters: {total:,}")
    typer.echo(f"Trainable parameters: {trainable:,}")


@app.command("train")
def train(
    config: Annotated[
        Path | None, typer.Option(help="Config YAML; default if unset.")
    ] = None,
    train_file: Annotated[
        Path | None,
        typer.Option(help="FI-2010 training matrix text file."),
    ] = None,
    val_file: Annotated[
        Path | None,
        typer.Option(help="Optional separate validation matrix."),
    ] = None,
    val_fraction: Annotated[
        float, typer.Option(help="Holdout fraction when no val file.")
    ] = 0.2,
    synthetic: Annotated[
        bool, typer.Option(help="Train on generated synthetic data.")
    ] = False,
    synthetic_events: Annotated[
        int, typer.Option(help="Events to synthesise when --synthetic.")
    ] = 20_000,
) -> None:
    """Train a model and checkpoint the best validation epoch."""
    experiment = _load_config(config)
    configure_logging(experiment.train.log_level)

    train_matrix = _resolve_training_matrix(
        train_file, synthetic, synthetic_events, experiment
    )
    if val_file is not None:
        val_matrix = load_lob_matrix(val_file)
    else:
        train_matrix, val_matrix = split_session(train_matrix, val_fraction)

    train_loader = make_loader(
        matrix_to_dataset(train_matrix, experiment.data),
        experiment.data,
        shuffle=experiment.data.shuffle_train,
    )
    val_loader = make_loader(
        matrix_to_dataset(val_matrix, experiment.data),
        experiment.data,
        shuffle=False,
    )

    trainer = Trainer.build(experiment)
    history = trainer.fit(train_loader, val_loader)
    best = min(history, key=lambda report: report.val_loss)
    typer.echo(
        f"Best epoch {best.epoch}: val_loss={best.val_loss:.4f} "
        f"val_acc={best.val_accuracy:.4f}"
    )
    if trainer.best_checkpoint_path is not None:
        typer.echo(f"Checkpoint saved to {trainer.best_checkpoint_path}")


@app.command("evaluate")
def evaluate(
    checkpoint: Annotated[
        Path, typer.Option(help="Checkpoint file to evaluate.")
    ],
    test_file: Annotated[
        Path | None, typer.Option(help="FI-2010 test matrix file.")
    ] = None,
    synthetic: Annotated[
        bool, typer.Option(help="Evaluate on generated synthetic data.")
    ] = False,
    synthetic_events: Annotated[
        int, typer.Option(help="Events to synthesise when --synthetic.")
    ] = 10_000,
) -> None:
    """Evaluate a checkpoint and print a classification report."""
    configure_logging()
    loaded = load_checkpoint(checkpoint)

    if synthetic:
        matrix = generate_lob_matrix(num_events=synthetic_events, seed=1)
    elif test_file is not None:
        matrix = load_lob_matrix(test_file)
    else:
        raise typer.BadParameter("Provide --test-file or pass --synthetic.")

    loader = make_loader(
        matrix_to_dataset(matrix, loaded.config.data),
        loaded.config.data,
        shuffle=False,
    )
    device = resolve_device(loaded.config.train.device)
    metrics = evaluate_classifier(loaded.model, loader, device)
    typer.echo(metrics.report)
    typer.echo(f"Accuracy: {metrics.accuracy:.4f}")
    typer.echo(f"Macro F1: {metrics.macro_f1:.4f}")


def _resolve_training_matrix(
    train_file: Path | None,
    synthetic: bool,
    synthetic_events: int,
    experiment: ExperimentConfig,
) -> NDArray[np.float64]:
    """Pick the training matrix source based on the supplied flags."""
    if synthetic:
        logger.info("Synthesising %d events for training.", synthetic_events)
        return generate_lob_matrix(
            num_events=synthetic_events,
            num_features=experiment.data.num_features,
            seed=experiment.train.seed,
        )
    if train_file is None:
        raise typer.BadParameter("Provide --train-file or pass --synthetic.")
    return load_lob_matrix(train_file)


if __name__ == "__main__":
    app()
