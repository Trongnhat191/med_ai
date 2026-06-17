"""
Configuration for MIC Prediction project.

Paths are resolved from environment variables so the same code works both
locally and on Google Colab without modifications.

Environment variables (all optional — defaults shown below):
    MIC_ROOT       : project root dir (default: parent of this file)
    MIC_DATA_DIR   : data directory   (default: MIC_ROOT/data)
    MIC_FNA_DIR    : raw .fna files   (default: MIC_DATA_DIR/fna_data/raw_data)
    MIC_DATA_MAP   : label CSV        (default: MIC_DATA_DIR/fna_data/data_map.csv)
    MIC_SMILES_CSV : SMILES CSV       (default: MIC_DATA_DIR/antibiotic_smiles_data/...)

Usage on Colab:
    import os
    os.environ["MIC_ROOT"] = "/content/drive/MyDrive/med_ai"
    import source.config as cfg  # picks up env vars
"""
import os
from pathlib import Path

# ---- Path resolution (env-var overridable) --------------------------------
def _p(env_key: str, default: Path) -> Path:
    val = os.environ.get(env_key)
    return Path(val) if val else default

ROOT      = _p("MIC_ROOT",       Path(__file__).parent.parent)
DATA_DIR  = _p("MIC_DATA_DIR",   ROOT / "data")
FNA_DIR   = _p("MIC_FNA_DIR",    DATA_DIR / "fna_data" / "raw_data")
DATA_MAP  = _p("MIC_DATA_MAP",   DATA_DIR / "fna_data" / "data_map.csv")
SMILES_CSV= _p("MIC_SMILES_CSV", DATA_DIR / "antibiotic_smiles_data"
                                           / "antibiotic_smiles_20_drugs.csv")

# ---- Genome encoder (NTv3) ------------------------------------------------
GENOME_MODEL   = "InstaDeepAI/NTv3_100M_pre"
TOP_N_CONTIGS  = 3
# 65 536 = 512 * 128  — safe on T4 16 GB with NTv3's U-Net downsampling
MAX_CONTIG_LEN = 65536
GENOME_EMB_DIM = 768

# ---- SMILES encoder (ChemBERTa) -------------------------------------------
SMILES_MODEL   = "seyonec/ChemBERTa-zinc-base-v1"
SMILES_EMB_DIM = 768
SMILES_MAX_LEN = 128

# ---- Fusion MLP ------------------------------------------------------------
FUSION_HIDDEN  = [512, 128]
DROPOUT        = 0.3

# ---- Classification bins ---------------------------------------------------
# MIC dilution series: 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, ≥128
# Class index i  ←→  MIC_BINS[i]
# Values outside the range are clipped to the nearest bin.
MIC_BINS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
N_CLASSES = len(MIC_BINS)  # 11

import math as _math

def mic_to_class(mic_value: float) -> int:
    """
    Map a numeric MIC value (mg/L) to its bin index (0 … N_CLASSES-1).

    Strategy: convert to log2, round to nearest integer, then clip to the
    valid range defined by MIC_BINS.
      MIC_BINS = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
      log2     = [  -3,   -2,  -1, 0, 1, 2, 3,  4,  5,  6,   7]
      class    = [   0,    1,   2, 3, 4, 5, 6,  7,  8,  9,  10]
    """
    log2_min = _math.log2(MIC_BINS[0])            # -3
    log2_max = _math.log2(MIC_BINS[N_CLASSES - 1]) #  7
    log2_val = _math.log2(max(mic_value, 1e-6))
    log2_clipped = max(log2_min, min(log2_max, log2_val))
    return int(round(log2_clipped - log2_min))     # 0-indexed

# ---- Training --------------------------------------------------------------
BATCH_SIZE      = 4      # conservative for T4 16 GB
LR              = 1e-4
EPOCHS          = 50
FREEZE_ENCODERS = True   # freeze pretrained encoders; only MLP is trained
DEVICE          = "cuda" # Colab T4 provides a CUDA GPU
SEED            = 42

