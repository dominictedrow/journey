"""Tokenizers: image [N, C, H, W] → patch tokens [N, (H/P)·(W/P), E] (02 §C1).

PatchEmbed = the original ViT tokenizer: cut the image into P×P patches, flatten each, multiply by one learned matrix.
A Conv2d with kernel = stride = P computes exactly that (tests/test_model.py proves it against unfold + Linear).

ConvStem (the default) = a few 3×3 stride-2 convs, then a 1×1 projection to E (Xiao et al. 2021, "Early convolutions
help transformers see better"). Same output grid as PatchEmbed(P=16), but the tokens are built from overlapping local
features, which makes AdamW training more stable and helps on small data [F §1, §7].

Norm choice in the stem: GroupNorm, not the paper's BatchNorm. [derived] Our micro-batches are 1–2 studies, and MAE
feeds images with 75% of the patches zeroed, so BatchNorm running statistics would differ between pretraining,
fine-tuning and eval; GroupNorm has no batch statistics. BatchNorm stays an ablation (norm="bn").
"""
from __future__ import annotations

import torch
from torch import nn


class PatchEmbed(nn.Module):
    """Patchify: Conv2d(C, E, kernel=P, stride=P) → flatten → [N, n_patches, E]."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 384, patch: int = 16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def grid(self, h: int, w: int) -> tuple[int, int]:
        return h // self.patch, w // self.patch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                      # [N, E, H/P, W/P]
        return x.flatten(2).transpose(1, 2)   # [N, H/P·W/P, E], row-major patch order


def _norm2d(kind: str, ch: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    if kind == "gn":
        return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
    raise ValueError(f"unknown stem norm {kind!r}")


class ConvStem(nn.Module):
    """log2(P) × [3×3 conv stride 2 → norm → GELU], widths E/2^k growing to E, then a 1×1 conv to E.

    Total stride = P, so the token grid matches PatchEmbed(P) and everything downstream is unchanged.
    """

    def __init__(self, in_ch: int = 3, embed_dim: int = 384, patch: int = 16, norm: str = "gn"):
        super().__init__()
        n = patch.bit_length() - 1
        if 2 ** n != patch or n < 1:
            raise ValueError(f"ConvStem needs a power-of-two patch size, got {patch}")
        self.patch = patch
        widths = [max(embed_dim // 2 ** (n - 1 - i), 16) for i in range(n)]  # e.g. E=384, P=16: 48, 96, 192, 384
        layers: list[nn.Module] = []
        c = in_ch
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, stride=2, padding=1, bias=False), _norm2d(norm, w), nn.GELU()]
            c = w
        layers.append(nn.Conv2d(c, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*layers)

    def grid(self, h: int, w: int) -> tuple[int, int]:
        return h // self.patch, w // self.patch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


def build_tokenizer(kind: str, in_ch: int, embed_dim: int, patch: int, stem_norm: str = "gn") -> nn.Module:
    if kind == "conv_stem":
        return ConvStem(in_ch, embed_dim, patch, stem_norm)
    if kind == "patchify":
        return PatchEmbed(in_ch, embed_dim, patch)
    raise ValueError(f"unknown tokenizer {kind!r}")
