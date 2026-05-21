# 01 - Synthetic quickstart

The fastest way to see DepthCast work end to end. This example:

1. generates a small synthetic FI-2010-shaped session,
2. splits it into training and validation,
3. trains a tiny model for two epochs, and
4. evaluates the best checkpoint and prints a classification report.

No dataset and no GPU are required; it runs in a few seconds on a CPU.

## Run it

```bash
cd examples/01_synthetic_quickstart
python run.py
```

## Expected output

You will see structured log lines for each epoch followed by accuracy,
macro F1, and a per-class report. Because the data is synthetic, the
numbers are only meaningful as a sanity check that the pipeline trains and
evaluates correctly - they are not indicative of real-market performance.

## Where to go next

- Swap the synthetic matrix for a real FI-2010 file (see
  [../02_train_fi2010](../02_train_fi2010)).
- Increase `conv_channels` / `fusion_channels` in the config for a larger
  model once you move to a GPU.
