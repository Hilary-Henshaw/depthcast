"""Benchmark the core DepthCast operations and print a Markdown table.

This script is app-independent: it builds a model in-process and times it.
It writes the result to stdout only (no file), so it composes with the
README injector::

    python scripts/benchmark.py | \\
        python scripts/inject_readme_section.py --section benchmarks

Run on its own to just see the numbers::

    python scripts/benchmark.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from depthcast.config import DataConfig, ModelConfig
from depthcast.data import generate_lob_matrix
from depthcast.models import DepthCastNet
from depthcast.pipeline import matrix_to_dataset

_WARMUP = 3
_ITERS = 20
_WINDOW = 100


def _time_ms(operation: object, iterations: int, warmup: int) -> list[float]:
    """Time a zero-argument callable, returning per-iteration ms."""
    assert callable(operation)
    for _ in range(warmup):
        operation()
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _summary(samples: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, p50, p95, p99) in milliseconds."""
    array = np.asarray(samples, dtype=np.float64)
    return (
        float(array.mean()),
        float(np.percentile(array, 50)),
        float(np.percentile(array, 95)),
        float(np.percentile(array, 99)),
    )


def _latency_rows(model: DepthCastNet) -> list[str]:
    """Benchmark forward and predict_proba at a few batch sizes."""
    rows: list[str] = []
    model.eval()
    for label, batch in (("forward", 1), ("forward", 32)):
        inputs = torch.randn(batch, 1, _WINDOW, model.config.num_features)

        def run(data: torch.Tensor = inputs) -> None:
            with torch.no_grad():
                model(data)

        mean, p50, p95, p99 = _summary(_time_ms(run, _ITERS, _WARMUP))
        throughput = batch / (mean / 1000.0)
        rows.append(
            f"| {label} | {batch} | {mean:.2f} | {p50:.2f} | "
            f"{p95:.2f} | {p99:.2f} | {throughput:,.0f} |"
        )

    proba_inputs = torch.randn(32, 1, _WINDOW, model.config.num_features)

    def run_proba(data: torch.Tensor = proba_inputs) -> None:
        model.predict_proba(data)

    mean, p50, p95, p99 = _summary(_time_ms(run_proba, _ITERS, _WARMUP))
    rows.append(
        f"| predict_proba | 32 | {mean:.2f} | {p50:.2f} | "
        f"{p95:.2f} | {p99:.2f} | {32 / (mean / 1000.0):,.0f} |"
    )
    return rows


def _windowing_row() -> str:
    """Benchmark lazy windowing build and iteration throughput."""
    events = 20_000
    matrix = generate_lob_matrix(num_events=events, seed=0)
    data = DataConfig(window=_WINDOW)

    start = time.perf_counter()
    dataset = matrix_to_dataset(matrix, data)
    build_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    for index in range(len(dataset)):
        dataset[index]
    iterate_s = time.perf_counter() - start
    rate = len(dataset) / iterate_s
    return (
        f"| windowing | {events:,} | {len(dataset):,} | "
        f"{build_ms:.2f} | {rate:,.0f} |"
    )


def main() -> None:
    """Run all benchmarks and print a Markdown report to stdout."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = DepthCastNet(ModelConfig())

    lines = [
        "**Model operation latency**",
        "",
        "| Operation | Batch | Mean (ms) | p50 (ms) | p95 (ms) | "
        "p99 (ms) | Throughput (samples/s) |",
        "|---|---|---|---|---|---|---|",
        *_latency_rows(model),
        "",
        "**Lazy windowing**",
        "",
        "| Operation | Events | Windows | Build (ms) | Iterate (windows/s) |",
        "|---|---|---|---|---|",
        _windowing_row(),
        "",
        f"_Measured on CPU, 1 thread, window {_WINDOW}, default model "
        "(32 conv / 64 fusion channels). Indicative only; rerun on your "
        "own hardware._",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
