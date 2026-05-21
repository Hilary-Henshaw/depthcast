# 03 - Checkpoint inference

Demonstrates DepthCast's portable checkpoints and single-window
inference. The script:

1. trains a tiny model on synthetic data and saves the best checkpoint,
2. reloads it with `load_checkpoint` (no model class needed at the call
   site beyond the library), inspecting the recorded epoch and config,
   and
3. runs `predict_proba` on one order-book window to get a probability
   distribution over the three movement classes.

This is the pattern you would use to serve a trained model: ship the
checkpoint file, reload it, and call `predict_proba`.

## Run it

```bash
cd examples/03_checkpoint_inference
python run.py
```

## Expected output

Log lines for training, then the reloaded checkpoint's metadata, then a
probability vector that sums to one for a single sampled window.
