"""
Fusion MLP: concatenate genome + SMILES embeddings → predict MIC class logits.

Architecture:
    concat([genome_emb, smiles_emb])   (B, 1536)
    → Linear(1024 → 512) + BN + ReLU + Dropout
    → Linear(512  → 128) + BN + ReLU + Dropout
    → Linear(128  → N_CLASSES)
    → logits per MIC bin               (B, N_CLASSES)
"""
from typing import List

import torch
import torch.nn as nn

from source.config import (
    GENOME_EMB_DIM,
    SMILES_EMB_DIM,
    FUSION_HIDDEN,
    DROPOUT,
    N_CLASSES,
)


class FusionMLP(nn.Module):
    """Concatenation-based fusion with a small MLP prediction head."""

    def __init__(self):
        super().__init__()
        in_dim = GENOME_EMB_DIM + SMILES_EMB_DIM  # 768 + 768 = 1536
        layers: List[nn.Module] = []
        prev = in_dim
        for hidden_dim in FUSION_HIDDEN:
            layers.extend([
                nn.Linear(prev, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=DROPOUT),
            ])
            prev = hidden_dim
        layers.append(nn.Linear(prev, N_CLASSES))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        genome_emb: torch.Tensor,
        smiles_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            genome_emb: (B, GENOME_EMB_DIM)
            smiles_emb: (B, SMILES_EMB_DIM)

        Returns:
            (B, N_CLASSES) — raw logits for each MIC bin class
        """
        fused = torch.cat([genome_emb, smiles_emb], dim=-1)  # (B, 1536)
        return self.mlp(fused)                                # (B, N_CLASSES)
