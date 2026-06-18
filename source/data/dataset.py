"""MICDataset: maps (genome_path, smiles, mic_class) triplets for training.

Supports two modes:
  - raw mode   : returns {fna_path, smiles, mic_class} — encoders run per batch
  - cached mode: returns {genome_emb, smiles_emb, mic_class} — loaded from .pt cache

Cached mode is automatically used when both cache files exist (see embed_cache.py).

MIC class labels (11 classes):
  class  0 → 0.125 mg/L
  class  1 → 0.25
  class  2 → 0.5
  class  3 → 1
  class  4 → 2
  class  5 → 4
  class  6 → 8
  class  7 → 16
  class  8 → 32
  class  9 → 64
  class 10 → ≥128
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from source.config import (
    DATA_MAP, FNA_DIR, SMILES_CSV, mic_to_class,
    GENOME_SOURCE, AMR_ALIGNED_DIR,
)
from source.data.smiles_loader import load_smiles_dict


def _default_cache_paths():
    """Return (genome_cache_path, smiles_cache_path) honouring env vars."""
    from source.config import DATA_DIR
    genome = Path(os.environ.get("MIC_GENOME_CACHE",
                                  DATA_DIR / "cache" / "genome_embs.pt"))
    smiles = Path(os.environ.get("MIC_SMILES_CACHE",
                                  DATA_DIR / "cache" / "smiles_embs.pt"))
    return genome, smiles


class MICDataset(Dataset):
    """
    Dataset of (genome_emb, smiles_emb, log2_mic) samples (cached mode)
    or (fna_path, smiles, log2_mic) samples (raw mode).

    Args:
        split:     "train" or "val"
        val_ratio: fraction of data to use for validation (default 0.1)
        seed:      random seed for shuffling
        genome_cache: path to genome_embs.pt (auto-detected if None)
        smiles_cache: path to smiles_embs.pt (auto-detected if None)
    """

    def __init__(
        self,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        genome_cache: Optional[Path] = None,
        smiles_cache: Optional[Path] = None,
    ):
        df = pd.read_csv(DATA_MAP)

        # Drop rows without a numeric MIC label
        df = df.dropna(subset=["Predicted MIC"])

        # Normalise PATRIC ID to string
        df["PATRIC ID"] = df["PATRIC ID"].astype(str).str.strip()

        # Build path column — handle float PATRIC IDs (e.g. 573.1322 → 573.13220)
        def _to_fna_path(patric_id: str) -> Path:
            candidate = FNA_DIR / f"{patric_id}.fna"
            return candidate

        df["fna_path"] = df["PATRIC ID"].apply(_to_fna_path)

        # Keep only rows whose genome file actually exists
        df = df[df["fna_path"].apply(lambda p: p.exists())].copy()

        # In 'amr' mode, also require the pre-aligned AMR gene file to exist
        if GENOME_SOURCE == "amr":
            def _has_amr(patric_id: str) -> bool:
                return (AMR_ALIGNED_DIR / patric_id).exists()
            df = df[df["PATRIC ID"].apply(_has_amr)].copy()

        # Load SMILES dict and filter to known antibiotics
        smiles_dict = load_smiles_dict(SMILES_CSV)
        df = df[df["Antibiotic"].isin(smiles_dict)].copy()
        df["smiles"] = df["Antibiotic"].map(smiles_dict)

        # Compute classification label (integer class index 0–9)
        df["mic_class"] = df["Predicted MIC"].astype(float).apply(mic_to_class)

        # Shuffle and split
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n_val = int(len(df) * val_ratio)
        if split == "train":
            self.df = df.iloc[n_val:].reset_index(drop=True)
        else:
            self.df = df.iloc[:n_val].reset_index(drop=True)

        # ---- Cache detection ------------------------------------------------
        g_path, s_path = _default_cache_paths()
        genome_cache = genome_cache or g_path
        smiles_cache = smiles_cache or s_path

        if genome_cache.exists() and smiles_cache.exists():
            self._genome_cache: Optional[dict] = torch.load(genome_cache, weights_only=True)
            self._smiles_cache: Optional[dict] = torch.load(smiles_cache, weights_only=True)
            self._use_cache = True
        else:
            self._genome_cache = None
            self._smiles_cache = None
            self._use_cache = False

    @property
    def use_cache(self) -> bool:
        return self._use_cache

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        mic_class = torch.tensor(row["mic_class"], dtype=torch.long)

        if self._use_cache:
            return {
                "genome_emb": self._genome_cache[str(row["fna_path"])],  # (256,)
                "smiles_emb": self._smiles_cache[row["smiles"]],          # (768,)
                "mic_class":  mic_class,
            }
        else:
            return {
                "fna_path":  str(row["fna_path"]),
                "smiles":    row["smiles"],
                "mic_class": mic_class,
            }
