"""Use the DepthCast library API directly, without training."""

from __future__ import annotations

import torch

from depthcast.config import ModelConfig
from depthcast.logging_utils import configure_logging, get_logger
from depthcast.models import DepthCastNet

logger = get_logger(__name__)


def main() -> None:
    """Build a model and run a forward and probability pass."""
    configure_logging()
    torch.set_num_threads(1)

    config = ModelConfig(conv_channels=16, fusion_channels=16)
    model = DepthCastNet(config)

    batch = torch.randn(8, 1, 100, config.num_features)
    logits = model(batch)
    logger.info("Logits shape: %s", tuple(logits.shape))

    probabilities = model.predict_proba(batch)
    logger.info("Probabilities shape: %s", tuple(probabilities.shape))
    row_sums = probabilities.sum(dim=1)
    logger.info(
        "All rows sum to one: %s",
        bool(torch.allclose(row_sums, torch.ones(8), atol=1e-5)),
    )

    total = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %d", total)


if __name__ == "__main__":
    main()
