"""Metrics for the NLP language-modelling domain."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def calculate_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean cross-entropy loss over a batch (token-level)."""
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
    return float(loss.item())


def calculate_perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute perplexity = exp(cross-entropy)."""
    ce = calculate_cross_entropy(logits, targets)
    return float(math.exp(ce))


def calculate_bpb(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute bits-per-byte from cross-entropy.

    Approximate: BPB = CE / ln(2). This is an approximation for
    character-level models where each token ≈ 1 byte on average.
    """
    ce = calculate_cross_entropy(logits, targets)
    return float(ce / math.log(2))
