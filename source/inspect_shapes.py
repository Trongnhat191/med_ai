"""
inspect_shapes.py — Dry-run script to verify all tensor shapes BEFORE training.

Loads both pretrained encoders, runs 2 sample genomes + SMILES through the
entire pipeline, and prints every intermediate tensor shape.

Usage:
    python -m source.inspect_shapes
"""
import torch

import source.config as cfg
from source.config import N_CLASSES, MIC_BINS
from source.data.genome_parser import get_top_contigs
from source.data.smiles_loader import load_smiles_dict
from source.models.genome_encoder import GenomeEncoder
from source.models.smiles_encoder import SMILESEncoder
from source.models.fusion import FusionMLP
from source.models.mic_predictor import MICPredictor

SEP = "=" * 65


def _fmt(label: str, shape) -> None:
    print(f"    {label:<45} {str(tuple(shape))}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{SEP}")
    print(f"  Device: {device}")
    print(SEP)

    # ------------------------------------------------------------------ #
    # Pick 2 sample genomes and 2 sample SMILES
    # ------------------------------------------------------------------ #
    fna_files = sorted(cfg.FNA_DIR.glob("*.fna"))
    if len(fna_files) < 2:
        raise RuntimeError(f"Need at least 2 .fna files in {cfg.FNA_DIR}")

    sample_paths = [str(fna_files[0]), str(fna_files[1])]
    smiles_dict  = load_smiles_dict(cfg.SMILES_CSV)
    sample_smiles = list(smiles_dict.values())[:2]
    sample_names  = list(smiles_dict.keys())[:2]

    print(f"\n  Sample genomes  : {[p.split('/')[-1] for p in sample_paths]}")
    print(f"  Sample SMILES   : {sample_names}")

    # ------------------------------------------------------------------ #
    # STEP 1 — Genome parsing
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  STEP 1 — Genome parsing (top-N contigs)")
    print(SEP)
    for path in sample_paths:
        contigs = get_top_contigs(
            cfg.FNA_DIR / path.split("/")[-1],
            cfg.TOP_N_CONTIGS,
            cfg.MAX_CONTIG_LEN,
        )
        print(f"\n  File: {path.split('/')[-1]}")
        print(f"    Contigs selected            : {len(contigs)}")
        for i, c in enumerate(contigs):
            print(f"    Contig {i+1} length (after trunc): {len(c):,} nt")

    # ------------------------------------------------------------------ #
    # STEP 2 — Genome encoder (NTv3)
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print(f"  STEP 2 — Genome Encoder   [{cfg.GENOME_MODEL}]")
    print(SEP)
    print("  Loading model (may download on first run)…")
    genome_enc = GenomeEncoder(freeze=True).to(device)

    # Show per-contig shape for first genome
    first_path = sample_paths[0]
    first_contigs = get_top_contigs(
        cfg.FNA_DIR / first_path.split("/")[-1],
        cfg.TOP_N_CONTIGS,
        cfg.MAX_CONTIG_LEN,
    )
    print(f"\n  Per-contig encoding (genome 1, {len(first_contigs)} contigs):")
    raw_embs = genome_enc._encode_contig_batch(first_contigs, device)
    _fmt("NTv3 output per contig (N_contigs, L, 256)", raw_embs.shape)

    print("\n  Batch encoding (B=2 genomes):")
    genome_emb = genome_enc(sample_paths, device)
    _fmt("genome_emb  →  (B, GENOME_EMB_DIM)", genome_emb.shape)

    # ------------------------------------------------------------------ #
    # STEP 3 — SMILES encoder (ChemBERTa)
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print(f"  STEP 3 — SMILES Encoder   [{cfg.SMILES_MODEL}]")
    print(SEP)
    print("  Loading model (may download on first run)…")
    smiles_enc = SMILESEncoder(freeze=True).to(device)
    smiles_emb = smiles_enc(sample_smiles, device)
    _fmt("smiles_emb  →  (B, SMILES_EMB_DIM)", smiles_emb.shape)

    # ------------------------------------------------------------------ #
    # STEP 4 — Fusion MLP
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  STEP 4 — Fusion MLP")
    print(SEP)
    fusion = FusionMLP().to(device)
    fused_input = torch.cat([genome_emb, smiles_emb], dim=-1)
    _fmt("concat input → (B, GENOME_EMB_DIM + SMILES_EMB_DIM)", fused_input.shape)
    logits = fusion(genome_emb, smiles_emb)
    _fmt(f"fusion output → (B, N_CLASSES={N_CLASSES})  [class logits]", logits.shape)
    pred_classes = logits.argmax(dim=-1).cpu().tolist()
    pred_mics    = [MIC_BINS[c] for c in pred_classes]
    print(f"\n    Predicted classes : {pred_classes}")
    print(f"    Predicted MIC bins: {pred_mics} mg/L")

    # ------------------------------------------------------------------ #
    # STEP 5 — Full model summary
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  STEP 5 — Full MICPredictor (end-to-end)")
    print(SEP)
    model = MICPredictor(freeze_encoders=True).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n    Total parameters     : {total_params:,}")
    print(f"    Trainable parameters : {trainable_params:,}  (encoders frozen)")

    final = model(sample_paths, sample_smiles, device)
    _fmt(f"final output → (B, N_CLASSES={N_CLASSES})  [class logits]", final.shape)
    final_classes = final.argmax(dim=-1).cpu().tolist()
    print(f"\n    Predicted classes : {final_classes}")
    print(f"    Predicted MIC bins: {[MIC_BINS[c] for c in final_classes]} mg/L")

    print(f"\n{'=' * 65}")
    print("  ✅  All shapes verified successfully!")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
