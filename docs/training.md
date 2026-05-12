# Training guide

This guide covers configuration, the training loop, checkpoints, and
reproducibility.

## Configuration model

Everything tunable lives in three Pydantic models under one
`ExperimentConfig`. Generate a starter file:

```bash
depthcast init-config my_run.yaml
```

### `DataConfig`

| Field | Default | Meaning |
|---|---|---|
| `window` | 100 | Snapshots per sample. Must exceed 18. |
| `horizon_index` | 4 | Index into `LABEL_HORIZONS` (0-4). |
| `num_features` | 40 | Feature columns. Must be divisible by 4. |
| `batch_size` | 64 | Windows per batch. |
| `num_workers` | 0 | DataLoader worker processes. |
| `shuffle_train` | true | Shuffle training windows each epoch. |

### `ModelConfig`

| Field | Default | Meaning |
|---|---|---|
| `conv_channels` | 32 | Channel width in the compression stack. |
| `fusion_channels` | 64 | Channels per fusion branch. |
| `lstm_hidden` | 64 | LSTM hidden size. |
| `lstm_layers` | 1 | Stacked LSTM layers. |
| `num_classes` | 3 | Output classes. |
| `leaky_slope` | 0.01 | Negative slope for LeakyReLU. |
| `recurrent_dropout` | 0.0 | Dropout between LSTM layers (needs >1). |
| `num_features` | 40 | Must match `DataConfig.num_features`. |

### `TrainConfig`

| Field | Default | Meaning |
|---|---|---|
| `epochs` | 50 | Maximum epochs. |
| `learning_rate` | 1e-4 | Adam learning rate. |
| `weight_decay` | 0.0 | Adam weight decay. |
| `grad_clip` | null | Max gradient norm, or null to disable. |
| `early_stop_patience` | 10 | Stop after N epochs without gain, or null. |
| `seed` | 42 | Seed for Python, NumPy, and PyTorch. |
| `device` | auto | `auto`, `cpu`, or `cuda`. |
| `checkpoint_dir` | checkpoints | Where `best.pt` is written. |
| `log_level` | INFO | Logging verbosity. |

Cross-field validation rejects mismatched feature counts and windows that
are too short for the convolutional stack, so misconfigurations fail
immediately with a clear message.

## The training loop

`Trainer` (`src/depthcast/training/trainer.py`) owns the optimiser and:

1. runs one optimisation pass per epoch (with optional gradient
   clipping),
2. evaluates loss and accuracy on the validation loader,
3. checkpoints whenever validation loss improves,
4. stops early once `early_stop_patience` epochs pass without
   improvement, and
5. returns a list of typed `EpochReport` records.

```python
from depthcast.training import Trainer

trainer = Trainer.build(config)   # seeds, then constructs the model
history = trainer.fit(train_loader, val_loader)
best = min(history, key=lambda r: r.val_loss)
```

`Trainer.build` seeds all random number generators **before** the model
is constructed, so weight initialisation is reproducible for a given
configuration.

## Checkpoints

Checkpoints store the model `state_dict` plus a JSON copy of the
configuration. Reload with:

```python
from depthcast.training import load_checkpoint

loaded = load_checkpoint("checkpoints/best.pt")
model = loaded.model           # already in eval mode
config = loaded.config
```

Loading uses PyTorch's `weights_only=True` path and rebuilds the
architecture from the recorded config, so it never unpickles arbitrary
objects.

## Choosing a device

Set `train.device` to `auto` (CUDA when available, else CPU), `cpu`, or
`cuda`. Requesting `cuda` on a host without a GPU raises a clear
`RuntimeError` rather than failing deep in the forward pass.

## Tips for a CPU-only machine

- Lower `conv_channels` and `fusion_channels` (for example to 8) for
  quick experiments; the default 32 channels are tuned for a GPU.
- Cap thread usage with `torch.set_num_threads(1)` if training competes
  with other work.
- Use the synthetic generator with a few thousand events to validate the
  pipeline before committing to a full FI-2010 run.
