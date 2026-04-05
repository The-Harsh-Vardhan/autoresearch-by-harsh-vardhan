"""Metrics for tabular classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy."""
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())


def calculate_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Cross-entropy loss scalar."""
    return float(F.cross_entropy(logits, targets).item())


def calculate_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Macro F1 score across all classes."""
    preds = logits.argmax(dim=-1)
    f1_sum = 0.0
    for c in range(num_classes):
        tp = float(((preds == c) & (targets == c)).sum().item())
        fp = float(((preds == c) & (targets != c)).sum().item())
        fn = float(((preds != c) & (targets == c)).sum().item())
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8) if (precision + recall) > 0 else 0.0
        f1_sum += f1
    return f1_sum / max(num_classes, 1)
