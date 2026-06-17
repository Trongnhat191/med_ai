"""Loss functions and evaluation metrics for MIC classification.

Task: predict MIC bin class (0-9) corresponding to
  [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, ≥128] mg/L

Metrics:
  - cross_entropy_loss   : standard multi-class CE
  - exact_accuracy       : fraction where pred class == true class
  - within_one_bin_acc   : fraction where |pred_class - true_class| <= 1
    (equivalent to "within-2-fold" accuracy in MIC dilution terms,
     since adjacent bins differ by a factor of 2)
"""
import torch
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits:  (B, N_CLASSES) — raw scores from FusionMLP
        targets: (B,)           — integer class indices (LongTensor)
    """
    return F.cross_entropy(logits, targets)


def exact_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of samples where predicted class == true class."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def within_one_bin_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Fraction of samples where |predicted_class - true_class| <= 1.
    Since each adjacent class is a 2× MIC dilution step, this is
    equivalent to "within-2-fold accuracy" used in AMR literature.
    """
    preds = logits.argmax(dim=-1)
    return ((preds - targets).abs() <= 1).float().mean().item()
