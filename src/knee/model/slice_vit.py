"""SliceViT: a 2D ViT over one 2.5D window (3 adjacent slices as channels) → one E-dim vector (02 §C1–C8).

    x [N, 3, H, W] ─tokenizer─▶ [N, n, E] ─+pos2d─▶ (+[CLS]) ─blocks─▶ LayerNorm ─pool─▶ [N, E]

n = (H/P)·(W/P) patch tokens (256 at H=256, P=16). pool="gap" (default) averages the patch tokens; pool="cls" prepends
a learned [CLS] token and returns its final state (Dom's list; GAP trains better from scratch [F §3]).

The pieces are exposed separately (`tokenize`, `run_blocks`, `pool`) because the MAE wrapper (knee/ssl/mae.py) runs
the same tokenizer and blocks on the visible tokens only.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from knee.model.block import Block
from knee.model.patch_embed import build_tokenizer
from knee.model.pos_embed import PosEmbed2D

# HP: pixel normalization of the uint8 cache. The cache is a per-series 1–99 percentile window, so pixels already span
# [0, 1] after /255; this only centres them. UNVERIFIED that the corpus mean is near 0.5 (S02 part 2 can measure it).
INPUT_MEAN, INPUT_STD = 0.5, 0.5


def to_input(x: torch.Tensor, mean: float = INPUT_MEAN, std: float = INPUT_STD) -> torch.Tensor:
    """uint8 pixels (any shape) → float32 normalized model input. Do this on the GPU (4× less to move)."""
    return (x.float() / 255.0 - mean) / std


def init_weights(m: nn.Module) -> None:
    """02 §C8: trunc_normal(0.02) Linear weights, zero biases, LayerNorm = identity."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class SliceViT(nn.Module):
    def __init__(self, in_ch: int = 3, img_size: int = 256, patch: int = 16, dim: int = 384, depth: int = 12,
                 heads: int = 6, mlp_ratio: float = 4.0, tokenizer: str = "conv_stem", stem_norm: str = "gn",
                 pos2d: str = "sincos", pool: str = "gap", qkv_bias: bool = True, qk_norm: bool = True,
                 layerscale: float | None = 0.1, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.1,
                 attn_impl: str = "sdpa", grad_checkpointing: bool = False):
        super().__init__()
        if pool not in ("gap", "cls"):
            raise ValueError(f"unknown pool {pool!r}")
        self.dim, self.depth, self.patch, self.pool_kind = dim, depth, patch, pool
        self.grad_checkpointing = grad_checkpointing
        self.tokenizer = build_tokenizer(tokenizer, in_ch, dim, patch, stem_norm)
        self.pos2d = PosEmbed2D(dim, pos2d, (img_size // patch, img_size // patch))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if pool == "cls" else None
        dpr = torch.linspace(0, drop_path, depth).tolist()  # stochastic depth grows linearly with depth
        self.blocks = nn.ModuleList([
            Block(dim, heads, mlp_ratio, qkv_bias, qk_norm, drop, attn_drop, dpr[i], layerscale, attn_impl)
            for i in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(init_weights)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=1e-6)  # near-zero init [F §5]

    # ---- pieces (shared with MAE)
    def tokenize(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """[N, C, H, W] → patch tokens + in-plane position [N, n, E], and the (gh, gw) grid. No [CLS] yet."""
        grid = self.tokenizer.grid(x.shape[-2], x.shape[-1])
        return self.pos2d(self.tokenizer(x), grid), grid

    def add_cls(self, t: torch.Tensor) -> torch.Tensor:
        if self.cls_token is None:
            return t
        return torch.cat([self.cls_token.expand(t.shape[0], -1, -1).to(t.dtype), t], dim=1)

    def run_blocks(self, t: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.grad_checkpointing and self.training and t.requires_grad:
                t = checkpoint(blk, t, use_reentrant=False)  # recompute in backward: less VRAM, ~30% more time
            else:
                t = blk(t)
        return self.norm(t)

    def pool(self, t: torch.Tensor) -> torch.Tensor:
        if self.pool_kind == "cls":
            return t[:, 0]
        return t.mean(dim=1)

    # ---- the standard path
    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """[N, C, H, W] → final normalized tokens [N, (1+)n, E]."""
        t, _ = self.tokenize(x)
        return self.run_blocks(self.add_cls(t))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[N, C, H, W] → [N, E] (one vector per window)."""
        return self.pool(self.forward_tokens(x))

    # ---- LLRD support (02 §C15)
    def layer_id(self, name: str) -> int:
        """0 = tokenizer/pos/cls, 1..depth = blocks, depth+1 = final norm. `name` is relative to this module."""
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        if name.startswith("norm."):
            return self.depth + 1
        return 0
