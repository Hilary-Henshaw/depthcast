# Architecture

DepthCast maps a window of limit-order-book snapshots to a three-class
prediction of the next mid-price move. This document walks through the
network layer by layer and explains why each stage exists.

## Input tensor

A single sample is a tensor of shape `(1, window, num_features)`:

- **channel = 1** so the order book can be read by a 2-D convolution,
- **window** consecutive events along the time axis (default 100),
- **num_features = 40**: ten order-book levels, each with an ask price,
  ask volume, bid price, and bid volume.

Batched, the input is `(batch, 1, window, 40)`.

## Stage 1: depth compression

`DepthCompressionStack` (`src/depthcast/models/blocks.py`) folds the 40
feature columns down to a single column over three stages. Every stage
ends with two height-4 convolutions that summarise local temporal
structure, so each stage removes 6 time steps; the three stages remove 18
in total.

| Stage | First conv | Effect on feature axis | Activation |
|---|---|---|---|
| `stage_pair` | `(1, 2)` stride `(1, 2)` | 40 -> 20 (splits price/volume) | LeakyReLU |
| `stage_level` | `(1, 2)` stride `(1, 2)` | 20 -> 10 (collapses level pairs) | Tanh |
| `stage_collapse` | `(1, 10)` | 10 -> 1 (collapses all levels) | LeakyReLU |

The width of the collapsing kernel is derived as
`num_features / 4`, so the stack generalises to any feature count
divisible by four rather than hard-coding the number ten.

Output shape: `(batch, conv_channels, window - 18, 1)`.

## Stage 2: multi-scale fusion

`MultiScaleFusion` reads the compressed sequence at three temporal scales
in parallel and concatenates them on the channel axis:

- a **short** branch: a `1x1` then a `3x1` convolution,
- a **long** branch: a `1x1` then a `5x1` convolution,
- a **pooled** branch: a `3x1` max-pool then a `1x1` convolution.

All branches use `padding="same"`, so the time axis is preserved. With
`fusion_channels = 64` the concatenated output has `192` channels.

Output shape: `(batch, 3 * fusion_channels, window - 18, 1)`.

## Stage 3: recurrent head

The fused tensor is permuted to `(batch, time, channels)` and fed to an
LSTM. The hidden state at the final time step is passed to a linear layer
that emits one logit per class.

```
permute (0, 2, 1, 3) -> squeeze -> (batch, window-18, 192)
LSTM(input=192, hidden=64) -> take last step -> (batch, 64)
Linear(64, 3) -> logits (batch, 3)
```

## Logits, not probabilities

`forward` returns raw logits. This is deliberate: it pairs correctly with
`torch.nn.CrossEntropyLoss`, which applies the log-softmax internally.
Call `DepthCastNet.predict_proba` when you actually want a probability
distribution; it applies the softmax once, under `torch.no_grad`, and
restores the module's previous train/eval mode.

## Shape cheat sheet

For the defaults (`window = 100`, `num_features = 40`,
`conv_channels = 32`, `fusion_channels = 64`):

```
(B, 1, 100, 40)
  -> compression -> (B, 32, 82, 1)
  -> fusion      -> (B, 192, 82, 1)
  -> reshape     -> (B, 82, 192)
  -> LSTM        -> (B, 64)
  -> head        -> (B, 3)
```

## Why these choices

- **Convolution before recurrence** lets the network learn local
  price/volume interactions cheaply before the more expensive sequential
  model reasons over time.
- **Multi-scale fusion** gives the recurrent layer features summarised at
  several horizons, which matters because order-book signals operate on
  different time scales.
- **A single LSTM layer** keeps the parameter count modest (about 144k
  parameters at the defaults) and trains stably on a single GPU.
