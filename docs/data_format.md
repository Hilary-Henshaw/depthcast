# Data format

DepthCast reads the FI-2010 benchmark and any data laid out the same way.
This document describes that layout and how DepthCast turns it into
supervised windows.

## The FI-2010 matrix

Each FI-2010 session is a dense numeric matrix stored as a whitespace
text file. **Rows are features; columns are order-book events.** The row
layout is:

```
row 0   ┐
...     │  40 limit-order-book feature rows
row 39  ┘  (10 levels x {ask price, ask volume, bid price, bid volume})
...        intermediate engineered features (ignored by DepthCast)
row -5  ┐
row -4  │
row -3  │  5 label rows, one per prediction horizon
row -2  │
row -1  ┘
```

DepthCast uses the first `num_features` rows (40 by default) as inputs and
the last five rows as labels. The five label rows correspond to the
horizons in `depthcast.constants.LABEL_HORIZONS`:

| Label row | Horizon (events) | `horizon_index` |
|---|---|---|
| -5 | 10 | 0 |
| -4 | 20 | 1 |
| -3 | 30 | 2 |
| -2 | 50 | 3 |
| -1 | 100 | 4 |

## Labels

Raw FI-2010 labels are one-based (`1`, `2`, `3`). DepthCast subtracts one
to produce zero-based class indices that align with
`depthcast.constants.MOVEMENT_CLASSES`:

| Class index | Meaning |
|---|---|
| 0 | up |
| 1 | stationary |
| 2 | down |

`split_features_labels` validates that every decoded label lands in
`[0, 2]` and raises a `ValueError` otherwise, which catches files that do
not follow the convention.

## From matrix to windows

Given a feature matrix of shape `(events, num_features)` and a label
vector of shape `(events,)`, `LobWindowDataset` produces overlapping
windows:

- window `j` covers events `[j, j + window)`,
- its label is the label aligned to the **last** event in the window,
- there are `events - window + 1` windows in total.

Windows are sliced on demand from the contiguous matrix rather than being
materialised up front, so memory use scales with the session length, not
with the number of windows.

## Obtaining FI-2010

The dataset is published by Ntakaris et al. and is free for research use.
Download the `NoAuction` decimal-precision files (`Train_Dst_*` and
`Test_Dst_*`) and pass them to the CLI or to `load_lob_matrix`. The files
are large (several gigabytes uncompressed), so they are excluded from
version control by `.gitignore`.

## No dataset? Use the synthetic generator

`depthcast.data.generate_lob_matrix` produces a deterministic matrix with
exactly this layout by simulating a drifting mid-price and reading
direction labels off it. It is the recommended source for tests,
examples, and quick smoke runs:

```python
from depthcast.data import generate_lob_matrix

matrix = generate_lob_matrix(num_events=20_000, seed=0)
# shape == (45, 20_000): 40 feature rows + 5 label rows
```
