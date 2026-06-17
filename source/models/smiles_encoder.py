"""
SMILES encoder using ChemBERTa (seyonec/ChemBERTa-zinc-base-v1).

Takes the [CLS] token from the last hidden state as the molecule embedding.
Output shape: (B, SMILES_EMB_DIM) = (B, 768)
"""
from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from source.config import SMILES_MODEL, SMILES_MAX_LEN


class SMILESEncoder(nn.Module):
    """Wraps ChemBERTa and produces a fixed-size drug embedding."""

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(SMILES_MODEL)
        self.model = AutoModel.from_pretrained(SMILES_MODEL)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, smiles_list: List[str], device: torch.device) -> torch.Tensor:
        """
        Args:
            smiles_list: list of B SMILES strings
            device:      torch device

        Returns:
            Tensor (B, SMILES_EMB_DIM) — [CLS] token embedding
        """
        batch = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=SMILES_MAX_LEN,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            out = self.model(**batch)

        # [CLS] token is at position 0
        cls_emb = out.last_hidden_state[:, 0, :]  # (B, 768)
        return cls_emb
