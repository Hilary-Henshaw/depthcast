# DepthCast examples

Five self-contained, runnable examples. None of them require the FI-2010
dataset except where explicitly noted; the rest use the built-in
synthetic generator so they run in seconds on a CPU.

| Example | What it shows |
|---|---|
| [01_synthetic_quickstart](01_synthetic_quickstart/) | End-to-end train and evaluate on synthetic data. |
| [02_train_fi2010](02_train_fi2010/) | Train on the real FI-2010 files. |
| [03_checkpoint_inference](03_checkpoint_inference/) | Save, reload, and run inference from a checkpoint. |
| [04_horizon_sweep](04_horizon_sweep/) | Compare prediction horizons. |
| [05_python_api](05_python_api/) | Use the library API without the CLI. |

## Running an example

From the repository root, with the package importable (either
`pip install -e .` or `PYTHONPATH=src`):

```bash
cd examples/01_synthetic_quickstart
python run.py
```

Each example directory has its own README explaining what it demonstrates
and how to run it.
