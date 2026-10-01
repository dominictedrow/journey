"""Position and type embeddings (02 §C2–C4).

- In-plane: `PosEmbed2D`, fixed 2D sin-cos by default (no parameters; regenerates for any grid, so MAE can pretrain at
  a lower resolution than fine-tuning). Learned is the ablation; it is resized by bicubic interpolation.
- Through-plane: `SlicePosEmbed`, the continuous position z ∈ [0,1] of a window's centre slice in its stack
  (+ optionally its distance in mm from the stack middle) → sin-cos features → Linear to E. Continuous because series
  have 11–320 slices and we sample sparse windows.
- Series type: `SeriesTypeEmbed`, a sum of learned lookups (plane, fluid, contrast class, field strength, slot).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from knee.data.series_select import CONTRASTS, PLANES, SLOTS

N_TYPE_IDS = {"plane": len(PLANES), "fluid": 2, "contrast": len(CONTRASTS), "field": 3, "slot": len(SLOTS) + 1}


def sincos_1d(pos: torch.Tensor, dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """pos [...] float → [..., dim] with dim/2 sines then dim/2 cosines (the Transformer formula)."""
    if dim % 2:
        raise ValueError("sincos_1d needs an even dim")
    omega = torch.arange(dim // 2, dtype=torch.float32, device=pos.device) / (dim // 2)
    omega = 1.0 / temperature ** omega
    ang = pos.float()[..., None] * omega
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def sincos_2d(gh: int, gw: int, dim: int, device=None) -> torch.Tensor:
    """[gh·gw, dim] for a row-major grid: half the channels encode the row, half the column (as in MAE)."""
    if dim % 4:
        raise ValueError("sincos_2d needs dim divisible by 4")
    ys, xs = torch.meshgrid(torch.arange(gh, dtype=torch.float32, device=device),
                            torch.arange(gw, dtype=torch.float32, device=device), indexing="ij")
    return torch.cat([sincos_1d(ys.flatten(), dim // 2), sincos_1d(xs.flatten(), dim // 2)], dim=-1)


class PosEmbed2D(nn.Module):
    """Adds a [1, gh·gw, E] in-plane position table to patch tokens [N, gh·gw, E]."""

    def __init__(self, dim: int, kind: str = "sincos", base_grid: tuple[int, int] = (16, 16)):
        super().__init__()
        if kind not in ("sincos", "learned"):
            raise ValueError(f"unknown pos2d {kind!r}")
        self.dim, self.kind, self.base_grid = dim, kind, tuple(base_grid)
        if kind == "learned":
            self.table = nn.Parameter(torch.zeros(1, base_grid[0] * base_grid[1], dim))
            nn.init.trunc_normal_(self.table, std=0.02)
        self._cache: dict = {}

    def table_for(self, grid: tuple[int, int], device) -> torch.Tensor:
        gh, gw = grid
        if self.kind == "sincos":
            key = (gh, gw, str(device))
            t = self._cache.get(key)
            if t is None:
                t = sincos_2d(gh, gw, self.dim, device)[None]
                self._cache[key] = t
            return t
        if (gh, gw) == self.base_grid:
            return self.table
        bh, bw = self.base_grid
        t = self.table.reshape(1, bh, bw, self.dim).permute(0, 3, 1, 2)
        t = F.interpolate(t, size=(gh, gw), mode="bicubic", align_corners=False)
        return t.permute(0, 2, 3, 1).reshape(1, gh * gw, self.dim)

    def forward(self, tokens: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        return tokens + self.table_for(grid, tokens.device).to(tokens.dtype)


class SlicePosEmbed(nn.Module):
    """Window tokens [B, L, E] + Linear(sincos(z) [, sincos(z_mm / mm_scale)]).

    z ∈ [0,1] is scaled by 2π·2^k for k = 0..n_freq-1, so the lowest frequency spans the whole stack once.
    The output Linear is zero-initialized: at the start of fine-tuning the MAE-pretrained window features are not
    perturbed, and the model learns how much slice position matters.
    """

    def __init__(self, dim: int, n_freq: int = 8, use_mm: bool = False, mm_scale: float = 100.0):
        super().__init__()
        self.n_freq, self.use_mm, self.mm_scale = n_freq, use_mm, mm_scale
        n_in = 2 * n_freq * (2 if use_mm else 1)
        self.proj = nn.Linear(n_in, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _feat(self, v: torch.Tensor) -> torch.Tensor:
        # built in fp32 on every call, not stored as a buffer: model.half() would round 2π·2^7 ≈ 804 to a step of 0.5
        freqs = 2 * math.pi * 2.0 ** torch.arange(self.n_freq, dtype=torch.float32, device=v.device)
        a = v.float()[..., None] * freqs
        return torch.cat([a.sin(), a.cos()], dim=-1)

    def forward(self, tokens: torch.Tensor, z: torch.Tensor, z_mm: torch.Tensor | None = None) -> torch.Tensor:
        f = self._feat(z)
        if self.use_mm:
            if z_mm is None:
                raise ValueError("SlicePosEmbed(use_mm=True) needs z_mm")
            f = torch.cat([f, self._feat(z_mm / self.mm_scale)], dim=-1)
        return tokens + self.proj(f.to(self.proj.weight.dtype)).to(tokens.dtype)


class SeriesTypeEmbed(nn.Module):
    """Window tokens [B, L, E] + Σ_k table_k[id_k] for the chosen fields (plane, fluid, contrast, field, slot).

    `series_idx` (0..max_series-1, the series' order in the bag) is an optional extra field so two series of the same
    type stay distinguishable (02 §C4 ablation).
    """

    def __init__(self, dim: int, fields: tuple[str, ...] = ("plane", "fluid", "contrast", "field"),
                 max_series: int = 0):
        super().__init__()
        unknown = set(fields) - set(N_TYPE_IDS)
        if unknown:
            raise ValueError(f"unknown series-type fields {unknown}")
        self.fields = tuple(fields)
        self.tables = nn.ModuleDict({k: nn.Embedding(N_TYPE_IDS[k], dim) for k in self.fields})
        if max_series:
            self.tables["series_idx"] = nn.Embedding(max_series, dim)
        for t in self.tables.values():
            nn.init.trunc_normal_(t.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, ids: dict[str, torch.Tensor]) -> torch.Tensor:
        out = tokens
        for k, table in self.tables.items():
            idx = ids[k]
            if k == "series_idx":
                idx = idx.clamp(max=table.num_embeddings - 1)
            out = out + table(idx).to(tokens.dtype)
        return out
