"""The transformer MLP: Linear(E, r·E) → GELU → Dropout → Linear(r·E, E) → Dropout (01 §2 #8)."""
from __future__ import annotations

import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, drop: float = 0.0, out_dim: int | None = None):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_dim or dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
