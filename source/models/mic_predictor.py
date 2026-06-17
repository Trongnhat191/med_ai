"""
Full dual-branch MIC predictor.

Forward pass:
    fna_paths  → GenomeEncoder  → (B, 768)
    smiles     → SMILESEncoder  → (B, 768)
                                → FusionMLP → (B,) log2(MIC)
"""
from typing import List

import torch
import torch.nn as nn

from source.config import FREEZE_ENCODERS
from source.models.genome_encoder import GenomeEncoder
from source.models.smiles_encoder import SMILESEncoder
from source.models.fusion import FusionMLP


class MICPredictor(nn.Module):
    """Dual-branch model: genome + drug → MIC (log2 scale)."""

    def __init__(self, freeze_encoders: bool = FREEZE_ENCODERS):
        super().__init__()
        self.genome_encoder = GenomeEncoder(freeze=freeze_encoders)
        self.smiles_encoder = SMILESEncoder(freeze=freeze_encoders)
        self.fusion = FusionMLP()

    def forward(
        self,
        fna_paths: List[str],
        smiles_list: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Args:
            fna_paths:   list of B absolute paths to .fna genome files
            smiles_list: list of B SMILES strings
            device:      torch device

        Returns:
            (B,) predicted log2(MIC)
        """
        genome_emb = self.genome_encoder(fna_paths, device)    # (B, GENOME_EMB_DIM)
        smiles_emb = self.smiles_encoder(smiles_list, device)  # (B, 768)
        return self.fusion(genome_emb, smiles_emb)             # (B,)
