"""Main training entry point.

Usage:
    python -m source.train
"""
import torch

import source.config as config
from source.data.dataset import MICDataset
from source.models.mic_predictor import MICPredictor
from source.training.trainer import run_training
from source.utils.logger import get_logger

logger = get_logger()


def main() -> None:
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    logger.info("Loading datasets…")
    train_ds = MICDataset(split="train", seed=config.SEED)
    val_ds   = MICDataset(split="val",   seed=config.SEED)
    logger.info(f"  Train samples : {len(train_ds)}")
    logger.info(f"  Val   samples : {len(val_ds)}")
    mode = "cached (fast)" if getattr(train_ds, "use_cache", False) else \
           "raw (slow) — run: python -m source.utils.embed_cache"
    logger.info(f"  Dataset mode  : {mode}")

    logger.info("Building model…")
    model = MICPredictor(freeze_encoders=config.FREEZE_ENCODERS)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable parameters: {trainable:,}")

    logger.info("Starting training…")
    run_training(model, train_ds, val_ds, config)


if __name__ == "__main__":
    main()
