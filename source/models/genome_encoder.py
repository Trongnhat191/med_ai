"""
Genome encoder using NTv3 pre-trained models
(e.g. InstaDeepAI/NTv3_8M_pre, InstaDeepAI/NTv3_100M_pre, InstaDeepAI/NTv3_650M_pre).

Two encoding modes are supported (controlled by config.GENOME_SOURCE):

  'fna'  — Whole-genome contig mode (original approach):
      - Parse top-N contigs from the .fna file
      - Tokenise + forward each contig through NTv3
      - Masked mean-pool over sequence length  → (GENOME_EMB_DIM,) per contig
      - Mean-pool over contigs                 → (GENOME_EMB_DIM,) per sample

  'amr'  — AMR gene slot mode (new, sharper signal):
      - Read the pre-aligned amr_genes_aligned file for this genome
      - Extract present AMR gene slots (non-N regions ≥ AMR_MIN_GENE_LEN nt)
      - Tokenise + forward all slots through NTv3
      - Masked mean-pool over sequence length  → (GENOME_EMB_DIM,) per slot
      - Mean-pool over all present slots       → (GENOME_EMB_DIM,) per sample

Both modes support a batched variant (embed_genomes_batched) that packs
sequences from multiple genomes into micro-batches for efficiency.
"""
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

from source.config import (
    GENOME_MODEL,
    GENOME_EMB_DIM,
    GENOME_SOURCE,
    TOP_N_CONTIGS,
    MAX_CONTIG_LEN,
    AMR_ALIGNED_DIR,
    AMR_MIN_GENE_LEN,
)
import warnings

from source.data.genome_parser import get_top_contigs
from source.data.amr_parser import get_amr_gene_slots, amr_path_for_genome


def _fna_to_patric_id(fna_path: str) -> str:
    """Extract PATRIC ID from an .fna path.  E.g. '/data/.../573.12862.fna' → '573.12862'."""
    return Path(fna_path).stem


class GenomeEncoder(nn.Module):
    """Wraps NTv3 pre-trained model and produces a fixed-size genome embedding.

    Supports two modes (set via config.GENOME_SOURCE):
      - 'fna': encode whole-genome contigs from .fna files (original behaviour)
      - 'amr': encode AMR gene slots from pre-aligned amr_genes_aligned files
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            GENOME_MODEL, trust_remote_code=True
        )
        self.ntv3 = AutoModelForMaskedLM.from_pretrained(
            GENOME_MODEL, trust_remote_code=True
        )
        if freeze:
            for param in self.ntv3.parameters():
                param.requires_grad = False

        self.mode = GENOME_SOURCE  # 'fna' or 'amr'

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_contig_batch(
        self, sequences: List[str], device: torch.device
    ) -> torch.Tensor:
        """
        Tokenise a list of DNA strings and run them through NTv3.

        Returns:
            Tensor of shape (len(sequences), GENOME_EMB_DIM) — masked mean-pooled
            over the sequence length dimension.
        """
        if not sequences:
            return torch.zeros(1, GENOME_EMB_DIM, device=device)

        batch = self.tokenizer(
            sequences,
            add_special_tokens=False,
            padding=True,
            pad_to_multiple_of=128,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=False,
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            out = self.ntv3(**batch, output_hidden_states=True, return_dict=True)

        # final hidden state: (N_seqs, L, GENOME_EMB_DIM)
        hidden = out.hidden_states[-1]

        # masked mean-pool over sequence length
        if "attention_mask" in batch:
            mask = batch["attention_mask"].unsqueeze(-1).float()  # (N, L, 1)
        else:
            mask = torch.ones(hidden.shape[0], hidden.shape[1], 1,
                              dtype=torch.float, device=device)
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return emb  # (N_seqs, GENOME_EMB_DIM)

    # ------------------------------------------------------------------
    # Single-genome embedding
    # ------------------------------------------------------------------

    def _get_sequences_fna(self, fna_path: str) -> List[str]:
        """Return top-N contigs from a .fna file (original mode)."""
        return get_top_contigs(Path(fna_path), TOP_N_CONTIGS, MAX_CONTIG_LEN)

    def _get_sequences_amr(self, fna_path: str) -> List[str]:
        """Return present AMR gene slots for a genome (new mode).

        The fna_path is used only to derive the PATRIC ID; the actual data is
        read from AMR_ALIGNED_DIR/<patric_id>.
        """
        patric_id = _fna_to_patric_id(fna_path)
        amr_file = amr_path_for_genome(patric_id, AMR_ALIGNED_DIR)
        if not amr_file.exists():
            warnings.warn(
                f"AMR aligned file not found for PATRIC ID '{patric_id}' "
                f"(expected: {amr_file}). "
                "Check that AMR_ALIGNED_DIR / MIC_AMR_ALIGNED env var is set correctly "
                "on Colab. Returning zero embedding for this genome.",
                stacklevel=2,
            )
            return []
        return get_amr_gene_slots(amr_file, min_len=AMR_MIN_GENE_LEN)

    def _get_sequences(self, fna_path: str) -> List[str]:
        """Dispatch to the correct sequence source based on self.mode."""
        if self.mode == "amr":
            return self._get_sequences_amr(fna_path)
        else:
            return self._get_sequences_fna(fna_path)

    def _embed_genome(self, fna_path: str, device: torch.device) -> torch.Tensor:
        """
        Parse a single genome and return a (1, GENOME_EMB_DIM) embedding.

        In 'amr' mode: encodes all present AMR gene slots.
        In 'fna' mode: encodes the top-N largest contigs.
        """
        sequences = self._get_sequences(fna_path)
        if not sequences:
            # Genome has no usable sequences → return zero vector
            return torch.zeros(1, GENOME_EMB_DIM, device=device)
        seq_embs = self._encode_contig_batch(sequences, device)  # (N, GENOME_EMB_DIM)
        return seq_embs.mean(dim=0, keepdim=True)               # (1, GENOME_EMB_DIM)

    # ------------------------------------------------------------------
    # Batched multi-genome embedding (efficient for cache building)
    # ------------------------------------------------------------------

    def embed_genomes_batched(
        self,
        fna_paths: List[str],
        device: torch.device,
        contig_batch_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Embed a list of genome paths efficiently by batching sequences across genomes.

        Instead of one GPU forward pass per genome, sequences from multiple genomes
        are packed into micro-batches of `contig_batch_size` and processed together.

        Works with both 'fna' and 'amr' modes — the sequence source differs but
        the batching logic is identical.

        In 'amr' mode, `contig_batch_size` controls how many AMR gene slots are
        processed together per forward pass.  On a T4 with NTv3_100M and typical
        AMR gene lengths (~500–2000 nt), 16–32 slots per micro-batch is safe.

        Args:
            fna_paths:         List of absolute paths to .fna files (PATRIC ID derived internally).
            device:            Torch device.
            contig_batch_size: Max number of sequences per forward pass.

        Returns:
            Dict mapping fna_path (str) → Tensor (GENOME_EMB_DIM,)
        """
        # Guard: nothing to encode
        if not fna_paths:
            return {}

        # 1. Collect all sequences and record which belong to which genome.
        #    We only add real sequences — empty genomes are handled at step 3.
        all_seqs: List[str] = []
        genome_slices: List[slice] = []   # slice into all_seqs for each genome

        for path in fna_paths:
            sequences = self._get_sequences(path)
            start = len(all_seqs)
            all_seqs.extend(sequences)   # extend with real seqs only (may be 0)
            genome_slices.append(slice(start, len(all_seqs)))

        # 2. Encode all sequences in micro-batches.
        #    If every genome returned [] (e.g. AMR files missing), all_seqs is
        #    empty and we skip this block entirely.
        all_emb_tensor: torch.Tensor | None = None
        if all_seqs:
            all_embs: List[torch.Tensor] = []
            for i in range(0, len(all_seqs), contig_batch_size):
                micro = all_seqs[i : i + contig_batch_size]
                emb = self._encode_contig_batch(micro, device)   # (k, GENOME_EMB_DIM)
                all_embs.append(emb.cpu())
            all_emb_tensor = torch.cat(all_embs, dim=0)  # (total_seqs, GENOME_EMB_DIM)

        # 3. Reconstruct per-genome embeddings.
        #    Genomes with no sequences get a zero vector.
        result: Dict[str, torch.Tensor] = {}
        for path, sl in zip(fna_paths, genome_slices):
            if all_emb_tensor is not None and (sl.stop - sl.start) > 0:
                result[path] = all_emb_tensor[sl].mean(dim=0)   # (GENOME_EMB_DIM,)
            else:
                result[path] = torch.zeros(GENOME_EMB_DIM)

        return result

    # ------------------------------------------------------------------
    # Forward (used in raw / non-cached training mode)
    # ------------------------------------------------------------------

    def forward(self, fna_paths: List[str], device: torch.device) -> torch.Tensor:
        """
        Args:
            fna_paths: list of B absolute paths to .fna files
            device:    torch device

        Returns:
            Tensor (B, GENOME_EMB_DIM)
        """
        embeddings = [self._embed_genome(p, device) for p in fna_paths]
        return torch.cat(embeddings, dim=0)  # (B, GENOME_EMB_DIM)
