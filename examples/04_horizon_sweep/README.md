# 04 - Horizon sweep

FI-2010 ships labels at five prediction horizons (10, 20, 30, 50, and 100
events ahead). This example trains a tiny model at several horizons on
synthetic data and compares validation accuracy, illustrating how
`horizon_index` changes the difficulty of the task.

Shorter horizons are noisier and generally harder to predict; longer
horizons aggregate more signal. The synthetic generator reflects this
because its labels are derived from future mid-price moves at each
horizon.

## Run it

```bash
cd examples/04_horizon_sweep
python run.py
```

## Expected output

One block of training logs per horizon, followed by a summary table of
validation accuracy by horizon. The absolute numbers are not meaningful
on synthetic data, but the relative trend across horizons is the point of
the example.
