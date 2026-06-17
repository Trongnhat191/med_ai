"""
Genome encoder using NTv3 pre-trained models
(e.g. InstaDeepAI/NTv3_8M_pre, InstaDeepAI/NTv3_100M_pre, InstaDeepAI/NTv3_650M_pre).

Strategy (single genome):
  - Parse top-N contigs from the .fna file
  - Tokenise + forward each contig through NTv3
  - Masked mean-pool over sequence length  → (GENOME_EMB_DIM,) per contig
  - Mean-pool over contigs                 → (GENOME_EMB_DIM,) per sample

Batched strategy (multiple genomes, faster for cache building):
  - Parse contigs from all genomes at once
  - Group contigs into contig-count-bounded micro-batches (avoids OOM)
  - Single forward pass per micro-batch
  - Reconstruct per-genome embeddings via index map
"""
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

from source.config import (
    GENOME_MODEL,
    GENOME_EMB_DIM,
    TOP_N_CONTIGS,
    MAX_CONTIG_LEN,
)
from source.data.genome_parser import get_top_contigs


class GenomeEncoder(nn.Module):
    """Wraps NTv3 pre-trained model and produces a fixed-size genome embedding."""

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

        # final hidden state: (N_contigs, L, GENOME_EMB_DIM)
        hidden = out.hidden_states[-1]

        # masked mean-pool over sequence length
        if "attention_mask" in batch:
            mask = batch["attention_mask"].unsqueeze(-1).float()  # (N, L, 1)
        else:
            mask = torch.ones(hidden.shape[0], hidden.shape[1], 1,
                              dtype=torch.float, device=device)
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return emb  # (N_contigs, GENOME_EMB_DIM)

    def _embed_genome(self, fna_path: str, device: torch.device) -> torch.Tensor:
        """
        Parse a single .fna file and return a (1, GENOME_EMB_DIM) embedding.
        """
        contigs = get_top_contigs(Path(fna_path), TOP_N_CONTIGS, MAX_CONTIG_LEN)
        contig_embs = self._encode_contig_batch(contigs, device)  # (N, GENOME_EMB_DIM)
        return contig_embs.mean(dim=0, keepdim=True)               # (1, GENOME_EMB_DIM)

    def embed_genomes_batched(
        self,
        fna_paths: List[str],
        device: torch.device,
        contig_batch_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Embed a list of genome paths efficiently by batching contigs across genomes.

        Instead of one GPU forward pass per genome, contigs from multiple genomes
        are packed into micro-batches of `contig_batch_size` contigs and processed
        together.  On a T4 with NTv3_100M and MAX_CONTIG_LEN=65536, a
        contig_batch_size of 4–8 is a safe default (adjust down if OOM).

        Args:
            fna_paths:         List of absolute paths to .fna files.
            device:            Torch device.
            contig_batch_size: Max number of contigs per forward pass.

        Returns:
            Dict mapping fna_path (str) → Tensor (GENOME_EMB_DIM,)
        """
        # 1. Parse all genomes and record which contigs belong to which genome
        all_contigs: List[str] = []
        genome_slices: List[slice] = []   # slice into all_contigs for each genome

        for path in fna_paths:
            contigs = get_top_contigs(Path(path), TOP_N_CONTIGS, MAX_CONTIG_LEN)
            start = len(all_contigs)
            all_contigs.extend(contigs if contigs else [""])  # guard empty genome
            genome_slices.append(slice(start, len(all_contigs)))

        # 2. Encode all contigs in micro-batches
        all_embs: List[torch.Tensor] = []
        for i in range(0, len(all_contigs), contig_batch_size):
            micro = all_contigs[i : i + contig_batch_size]
            emb = self._encode_contig_batch(micro, device)   # (k, GENOME_EMB_DIM)
            all_embs.append(emb.cpu())

        all_emb_tensor = torch.cat(all_embs, dim=0)  # (total_contigs, GENOME_EMB_DIM)

        # 3. Reconstruct per-genome embeddings
        result: Dict[str, torch.Tensor] = {}
        for path, sl in zip(fna_paths, genome_slices):
            result[path] = all_emb_tensor[sl].mean(dim=0)   # (GENOME_EMB_DIM,)

        return result

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
