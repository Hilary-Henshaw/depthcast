"""Tests for device resolution and seeding."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from depthcast.runtime import resolve_device, seed_everything


def test_resolve_cpu_device() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_auto_falls_back_to_cpu_without_cuda() -> None:
    device = resolve_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device.type == expected


def test_resolve_unknown_device_raises() -> None:
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="CUDA is present on this host."
)
def test_requesting_cuda_without_hardware_raises() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_seed_everything_is_reproducible() -> None:
    seed_everything(123)
    first_np = np.random.rand(3)
    first_torch = torch.rand(3)

    seed_everything(123)
    assert np.allclose(first_np, np.random.rand(3))
    assert torch.allclose(first_torch, torch.rand(3))


def test_seed_everything_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)
