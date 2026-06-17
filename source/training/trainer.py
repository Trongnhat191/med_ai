"""Training and evaluation loops for MICPredictor.

Two training modes:
  - Cached mode (fast): dataset returns pre-computed embedding tensors.
    Only the FusionMLP runs during training — ~50-100× faster on Colab T4.
  - Raw mode (slow):    dataset returns fna_paths + smiles strings.
    Full model (NTv3 + ChemBERTa + FusionMLP) runs per batch.

Run `python -m source.utils.embed_cache` once to enable cached mode.
"""
import torch
from torch.utils.data import DataLoader

from source.training.losses import cross_entropy_loss, exact_accuracy, within_one_bin_accuracy
from source.utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def _collate_cached(batch):
    """For cached mode: stacks pre-computed embedding tensors."""
    return {
        "genome_emb": torch.stack([b["genome_emb"] for b in batch]),  # (B, 256)
        "smiles_emb": torch.stack([b["smiles_emb"] for b in batch]),  # (B, 768)
        "mic_class":  torch.stack([b["mic_class"]  for b in batch]),  # (B,) long
    }


def _collate_raw(batch):
    """For raw mode: collects paths/strings as lists."""
    return {
        "fna_paths": [b["fna_path"]  for b in batch],
        "smiles":    [b["smiles"]    for b in batch],
        "mic_class": torch.stack([b["mic_class"] for b in batch]),  # (B,) long
    }


# ---------------------------------------------------------------------------
# Fast cached epoch loops (FusionMLP only)
# ---------------------------------------------------------------------------

def _train_epoch_cached(fusion, loader, optimizer, device, epoch: int) -> dict:
    """Train one epoch (cached mode). Logs every step; returns epoch metrics."""
    fusion.train()
    total_loss = 0.0
    all_logits, all_targets = [], []
    n_steps = len(loader)

    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad()
        genome_emb = batch["genome_emb"].to(device)
        smiles_emb = batch["smiles_emb"].to(device)
        logits = fusion(genome_emb, smiles_emb)    # (B, N_CLASSES)
        target = batch["mic_class"].to(device)     # (B,) long
        loss   = cross_entropy_loss(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_targets.append(target.cpu())

    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    return {
        "train_loss":    total_loss / n_steps,
        "train_raw_acc": exact_accuracy(all_logits, all_targets),
        "train_1tier":   within_one_bin_accuracy(all_logits, all_targets),
    }



def _eval_epoch_cached(fusion, loader, device) -> dict:
    fusion.eval()
    all_logits, all_target = [], []
    with torch.no_grad():
        for batch in loader:
            genome_emb = batch["genome_emb"].to(device)
            smiles_emb = batch["smiles_emb"].to(device)
            logits = fusion(genome_emb, smiles_emb)
            all_logits.append(logits.cpu())
            all_target.append(batch["mic_class"])
    logits  = torch.cat(all_logits)
    targets = torch.cat(all_target)
    return {
        "val_loss": cross_entropy_loss(logits, targets).item(),
        "val_acc":  exact_accuracy(logits, targets),
        "val_w1b":  within_one_bin_accuracy(logits, targets),
    }


# ---------------------------------------------------------------------------
# Slow raw epoch loops (full model including encoders)
# ---------------------------------------------------------------------------

def _train_epoch_raw(model, loader, optimizer, device, epoch: int) -> dict:
    """Train one epoch (raw mode). Logs every step; returns epoch metrics."""
    model.train()
    total_loss = 0.0
    all_logits, all_targets = [], []
    n_steps = len(loader)

    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad()
        logits = model(batch["fna_paths"], batch["smiles"], device)  # (B, N_CLASSES)
        target = batch["mic_class"].to(device)                        # (B,) long
        loss   = cross_entropy_loss(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_targets.append(target.cpu())

    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    return {
        "train_loss":    total_loss / n_steps,
        "train_raw_acc": exact_accuracy(all_logits, all_targets),
        "train_1tier":   within_one_bin_accuracy(all_logits, all_targets),
    }


def _eval_epoch_raw(model, loader, device) -> dict:
    model.eval()
    all_logits, all_target = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["fna_paths"], batch["smiles"], device)
            all_logits.append(logits.cpu())
            all_target.append(batch["mic_class"])
    logits  = torch.cat(all_logits)
    targets = torch.cat(all_target)
    return {
        "val_loss": cross_entropy_loss(logits, targets).item(),
        "val_acc":  exact_accuracy(logits, targets),
        "val_w1b":  within_one_bin_accuracy(logits, targets),
    }


# ---------------------------------------------------------------------------
# Shared logging / checkpoint loop
# ---------------------------------------------------------------------------

def _run_loop(train_fn, eval_fn, train_loader, val_loader, optimizer,
              scheduler, model_to_save, config, device):
    best_val_loss = float("inf")
    for epoch in range(1, config.EPOCHS + 1):
        # ---- Train ----
        train_metrics = train_fn(train_loader, optimizer, device, epoch)
        # ---- Validate ----
        val_metrics   = eval_fn(val_loader, device)
        scheduler.step()

        # Epoch-end summary
        logger.info(
            f"\nEpoch {epoch:03d} SUMMARY\n"
            f"  Train | loss={train_metrics['train_loss']:.4f} "
            f"| raw_acc={train_metrics['train_raw_acc']:.3f} "
            f"| 1tier_acc={train_metrics['train_1tier']:.3f}\n"
            f"  Val   | loss={val_metrics['val_loss']:.4f} "
            f"| raw_acc={val_metrics['val_acc']:.3f} "
            f"| 1tier_acc={val_metrics['val_w1b']:.3f}"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(model_to_save.state_dict(), "best_mic_predictor.pt")
            logger.info("  → Saved best model checkpoint.")



# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_training(model, train_ds, val_ds, config):
    """
    Args:
        model:    MICPredictor instance
        train_ds: training MICDataset
        val_ds:   validation MICDataset
        config:   source.config module
    """
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")
    model = model.to(device)

    use_cache = getattr(train_ds, "use_cache", False)

    if use_cache:
        logger.info(
            "Embedding cache detected → fast training mode "
            "(FusionMLP only, encoders frozen)."
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=_collate_cached,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE * 4,
            shuffle=False,
            collate_fn=_collate_cached,
            num_workers=4,
            pin_memory=True,
        )
        fusion = model.fusion
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, fusion.parameters()),
            lr=config.LR,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS
        )
        _run_loop(
            train_fn=lambda loader, opt, dev, epoch: _train_epoch_cached(fusion, loader, opt, dev, epoch),
            eval_fn=lambda loader, dev: _eval_epoch_cached(fusion, loader, dev),
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model_to_save=model,
            config=config,
            device=device,
        )

    else:
        logger.warning(
            "No embedding cache found → slow training mode (encoders run every batch). "
            "Run  python -m source.utils.embed_cache  once for ~50× speedup."
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=_collate_raw,
            num_workers=0,   # keep model in main process
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=_collate_raw,
            num_workers=0,
            pin_memory=False,
        )
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LR,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS
        )
        _run_loop(
            train_fn=lambda loader, opt, dev, epoch: _train_epoch_raw(model, loader, opt, dev, epoch),
            eval_fn=lambda loader, dev: _eval_epoch_raw(model, loader, dev),
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model_to_save=model,
            config=config,
            device=device,
        )
