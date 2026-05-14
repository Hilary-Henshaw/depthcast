# Changelog

All notable changes to DepthCast are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Multi-horizon training that predicts every FI-2010 horizon jointly.
- Optional attention pooling over the recurrent outputs.
- Streaming inference helper for live order-book feeds.

## [0.1.0] - 2026-06-02

### Added
- `DepthCastNet`, a convolutional-recurrent classifier for limit-order-book
  price direction, composed from `DepthCompressionStack` and
  `MultiScaleFusion` blocks.
- Typed, validated configuration (`DataConfig`, `ModelConfig`,
  `TrainConfig`, `ExperimentConfig`) with YAML round-tripping.
- Memory-frugal `LobWindowDataset` that slices overlapping windows on
  demand instead of materialising them.
- FI-2010 reader and a deterministic synthetic session generator.
- `Trainer` with early stopping, gradient clipping, and best-epoch
  checkpointing.
- Portable, weights-only checkpoints that store a JSON copy of the
  configuration and reload through PyTorch's safe path.
- `evaluate_classifier` returning a typed metric summary.
- A `typer` command-line interface: `init-config`, `synth`, `summary`,
  `train`, and `evaluate`.
- Structured logging, full type hints, a pytest suite, and a GitHub
  Actions CI pipeline.

[Unreleased]: https://github.com/Hilary-Henshaw/depthcast/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Hilary-Henshaw/depthcast/releases/tag/v0.1.0
