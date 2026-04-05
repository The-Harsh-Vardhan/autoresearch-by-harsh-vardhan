"""Models for tabular classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class LogisticBaseline(nn.Module):
    """Single-layer logistic regression baseline."""

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SmallMLP(nn.Module):
    """Two hidden-layer MLP with dropout and ReLU."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
