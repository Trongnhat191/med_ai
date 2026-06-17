"""
Pre-compute genome and SMILES embeddings and save them to disk.

Run ONCE before training:
    python -m source.utils.embed_cache

This creates two cache files (controlled by config):
    MIC_GENOME_CACHE  (default: data/cache/genome_embs.pt)
    MIC_SMILES_CACHE  (default: data/cache/smiles_embs.pt)

Each cache is a dict:
    genome_embs.pt  →  { fna_path_str: Tensor(GENOME_EMB_DIM) }
    smiles_embs.pt  →  { smiles_str:   Tensor(SMILES_EMB_DIM) }

Subsequent training runs will load from cache — encoders are never called
during the training loop, making epochs ~50-100× faster on Colab T4.
"""
from __future__ import annotations

import argparse

import os
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

import source.config as config
from source.data.dataset import MICDataset
from source.models.genome_encoder import GenomeEncoder
from source.models.smiles_encoder import SMILESEncoder
from source.utils.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Cache paths (env-var overridable)
# ---------------------------------------------------------------------------

def _cache_path(env_key: str, default: Path) -> Path:
    val = os.environ.get(env_key)
    return Path(val) if val else default

GENOME_CACHE = _cache_path(
    "MIC_GENOME_CACHE",
    config.DATA_DIR / "cache" / "genome_embs.pt",
)
SMILES_CACHE = _cache_path(
    "MIC_SMILES_CACHE",
    config.DATA_DIR / "cache" / "smiles_embs.pt",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_unique(train_ds: MICDataset, val_ds: MICDataset):
    """Return sorted lists of unique fna_paths and unique smiles strings.

    Reads directly from the underlying DataFrame so this works in both
    raw mode *and* cached mode (where __getitem__ does not return fna_path).
    """
    fna_paths: set[str] = set()
    smiles_set: set[str] = set()
    for ds in (train_ds, val_ds):
        fna_paths.update(ds.df["fna_path"].astype(str).tolist())
        smiles_set.update(ds.df["smiles"].tolist())
    return sorted(fna_paths), sorted(smiles_set)


def build_genome_cache(
    fna_paths: List[str],
    encoder: GenomeEncoder,
    device: torch.device,
    out_path: Path,
    contig_batch_size: int = 8,
) -> Dict[str, torch.Tensor]:
    """Encode all genomes and write cache to disk.

    Args:
        fna_paths:         Unique genome paths to encode.
        encoder:           GenomeEncoder (frozen NTv3).
        device:            Torch device.
        out_path:          Where to save the .pt cache file.
        contig_batch_size: Number of contigs per GPU forward pass.
                           This is the only knob that affects VRAM.
                           Reduce if OOM (default 8 works on T4 + NTv3_100M).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Pre-computing genome embeddings for {len(fna_paths)} unique genomes "
        f"(contig_batch_size={contig_batch_size})…"
    )
    cache = encoder.embed_genomes_batched(
        fna_paths, device, contig_batch_size=contig_batch_size
    )
    torch.save(cache, out_path)
    logger.info(f"  Saved genome cache → {out_path}")
    return cache




def build_smiles_cache(
    smiles_list: List[str],
    encoder: SMILESEncoder,
    device: torch.device,
    out_path: Path,
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """Encode all SMILES strings in batches and write cache to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, torch.Tensor] = {}
    logger.info(f"Pre-computing SMILES embeddings for {len(smiles_list)} unique drugs…")
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="SMILES cache"):
        batch = smiles_list[i : i + batch_size]
        with torch.no_grad():
            embs = encoder(batch, device)   # (B, 768)
        for smi, emb in zip(batch, embs):
            cache[smi] = emb.cpu()          # (768,)
    torch.save(cache, out_path)
    logger.info(f"  Saved SMILES cache → {out_path}")
    return cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="Delete and rebuild existing caches (required after changing the genome model).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load datasets to discover all unique paths/SMILES
    logger.info("Scanning datasets for unique inputs…")
    train_ds = MICDataset(split="train", seed=config.SEED)
    val_ds   = MICDataset(split="val",   seed=config.SEED)
    fna_paths, smiles_list = _collect_unique(train_ds, val_ds)
    logger.info(f"  Unique genomes : {len(fna_paths)}")
    logger.info(f"  Unique drugs   : {len(smiles_list)}")

    # ---- Genome cache --------------------------------------------------------
    if GENOME_CACHE.exists() and not args.force:
        logger.warning(
            f"Genome cache already exists at {GENOME_CACHE} — skipping.\n"
            "  If you changed the genome model, re-run with --force to rebuild."
        )
    else:
        if GENOME_CACHE.exists():
            logger.info(f"--force: deleting old genome cache at {GENOME_CACHE}")
            GENOME_CACHE.unlink()
        logger.info("Loading GenomeEncoder (NTv3)…")
        genome_enc = GenomeEncoder(freeze=True).to(device)
        genome_enc.eval()
        build_genome_cache(fna_paths, genome_enc, device, GENOME_CACHE)
        del genome_enc          # free GPU memory before SMILES pass
        torch.cuda.empty_cache()

    # ---- SMILES cache --------------------------------------------------------
    if SMILES_CACHE.exists() and not args.force:
        logger.info(f"SMILES cache already exists at {SMILES_CACHE} — skipping.")
    else:
        if SMILES_CACHE.exists():
            logger.info(f"--force: deleting old SMILES cache at {SMILES_CACHE}")
            SMILES_CACHE.unlink()
        logger.info("Loading SMILESEncoder (ChemBERTa)…")
        smiles_enc = SMILESEncoder(freeze=True).to(device)
        smiles_enc.eval()
        build_smiles_cache(smiles_list, smiles_enc, device, SMILES_CACHE)
        del smiles_enc
        torch.cuda.empty_cache()

    logger.info("Cache build complete. You can now run training.")


if __name__ == "__main__":
    main()
