"""Pre-norm transformer blocks (02 §C7).

    Block:      x = x + DropPath(LayerScale(Attn(LN(x))))
                x = x + DropPath(LayerScale(MLP(LN(x))))
    CrossBlock: q = q + DropPath(LayerScale(CrossAttn(LN(q), LN_ctx(ctx))))
                q = q + DropPath(LayerScale(MLP(LN(q))))

Pre-norm: the residual stream x is never normalized, only the branch inputs are. Gradients flow straight down the
identity path, so deep stacks train without warmup tricks (post-norm, the original Transformer, is fragile).
LayerScale: each branch output is multiplied by a learned per-channel vector initialized small (HP ≈ 0.1), so each
block starts close to the identity. DropPath (stochastic depth): in training, drop a sample's whole branch with
probability p and rescale the survivors by 1/(1-p); identity in eval.
"""
from __future__ import annotations

import torch
from torch import nn

from knee.model.attention import CrossAttention, MultiHeadSelfAttention
from knee.model.mlp import Mlp


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)  # one draw per sample
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"p={self.p}"


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float | None = 0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init))) if init else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.gamma is None else x * self.gamma


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, qk_norm: bool = False,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0, layerscale: float | None = 0.1,
                 attn_impl: str = "sdpa"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(dim, heads, qkv_bias, qk_norm, attn_drop, drop, attn_impl)
        self.ls1 = LayerScale(dim, layerscale)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio, drop)
        self.ls2 = LayerScale(dim, layerscale)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), key_mask)))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


class CrossBlock(nn.Module):
    """Queries read from a context (window tokens); the context itself is not updated."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, qk_norm: bool = False,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0, layerscale: float | None = 0.1,
                 attn_impl: str = "sdpa"):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_ctx = nn.LayerNorm(dim, eps=1e-6)
        self.attn = CrossAttention(dim, heads, qkv_bias, qk_norm, attn_drop, drop, attn_impl)
        self.ls1 = LayerScale(dim, layerscale)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio, drop)
        self.ls2 = LayerScale(dim, layerscale)
        self.drop_path = DropPath(drop_path)

    def forward(self, q: torch.Tensor, ctx: torch.Tensor, key_mask: torch.Tensor | None = None,
                return_attn: bool = False):
        r = self.attn(self.norm_q(q), self.norm_ctx(ctx), key_mask, return_attn)
        a, attn = r if return_attn else (r, None)
        q = q + self.drop_path(self.ls1(a))
        q = q + self.drop_path(self.ls2(self.mlp(self.norm2(q))))
        return (q, attn) if return_attn else q
