"""Multi-head attention, written out by hand and then swapped for PyTorch's fused kernel (02 §C6).

Shapes (B = batch of sequences, L = tokens, E = embedding dim, h = heads, d = E / h):

    x [B, L, E] ──qkv Linear──▶ [B, L, 3E] ──split──▶ q, k, v each [B, h, L, d]
    scores = q @ kᵀ / √d            [B, h, L, L]   (how much token i looks at token j, per head)
    scores[..., j] = -inf where key j is padding
    attn  = softmax(scores, dim=-1) [B, h, L, L]   (each row sums to 1 over the real keys)
    out   = attn @ v                [B, h, L, d] ──merge heads──▶ [B, L, E] ──proj Linear──▶ [B, L, E]

impl="manual" runs exactly those lines (readable, returns the attention map for plots).
impl="sdpa" calls F.scaled_dot_product_attention (FlashAttention / memory-efficient kernels on CUDA): same maths,
never materializes the L×L matrix. tests/test_model.py checks manual ≡ sdpa ≡ nn.MultiheadAttention.

Mask convention everywhere in this package: **key_mask True = a real token, False = padding** (the same as the
datasets' window_mask). Note that nn.MultiheadAttention uses the opposite (True = ignore).

QK-norm (optional, on by default in configs): a LayerNorm over d on q and k before the dot product. It bounds the
logits, which keeps bf16 training stable [F §4].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _safe_key_mask(key_mask: torch.Tensor) -> torch.Tensor:
    """A sequence with no real key would give softmax(all -inf) = NaN. Unmask key 0 there; its output is padding
    anyway and is ignored downstream, but NaNs would poison the gradients of the whole batch."""
    empty = ~key_mask.any(dim=-1)
    if empty.any():
        key_mask = key_mask.clone()
        key_mask[empty, 0] = True
    return key_mask


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_mask: torch.Tensor | None = None,
           impl: str = "sdpa", dropout_p: float = 0.0, return_attn: bool = False):
    """q [B, h, Lq, d], k/v [B, h, Lk, d], key_mask [B, Lk] (True = real) → out [B, h, Lq, d] (, attn [B, h, Lq, Lk])."""
    if key_mask is not None:
        key_mask = _safe_key_mask(key_mask)
    if impl == "sdpa" and not return_attn:
        m = None if key_mask is None else key_mask[:, None, None, :]  # bool, True = may attend
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m, dropout_p=dropout_p)
    if impl not in ("manual", "sdpa"):
        raise ValueError(f"unknown attention impl {impl!r}")
    scale = q.shape[-1] ** -0.5
    scores = (q * scale) @ k.transpose(-2, -1)                          # [B, h, Lq, Lk]
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
    attn = scores.float().softmax(dim=-1).to(q.dtype)                    # softmax in fp32 for bf16 safety
    out = F.dropout(attn, p=dropout_p) @ v
    return (out, attn) if return_attn else out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, qkv_bias: bool = True, qk_norm: bool = False, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, impl: str = "sdpa"):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads, self.head_dim, self.impl = heads, dim // heads, impl
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None, return_attn: bool = False):
        B, L, E = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, h, L, d]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        r = attend(q, k, v, key_mask, self.impl, self.attn_drop if self.training else 0.0, return_attn)
        out, attn = r if return_attn else (r, None)
        out = self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, L, E)))  # merge heads: [B, L, h·d]
        return (out, attn) if return_attn else out


class CrossAttention(nn.Module):
    """Queries [B, Q, E] attend over context tokens [B, L, E] (the finding-query head: 12 queries ← window tokens)."""

    def __init__(self, dim: int, heads: int, qkv_bias: bool = True, qk_norm: bool = False, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, impl: str = "sdpa"):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads, self.head_dim, self.impl = heads, dim // heads, impl
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, queries: torch.Tensor, context: torch.Tensor, key_mask: torch.Tensor | None = None,
                return_attn: bool = False):
        B, Q, E = queries.shape
        L = context.shape[1]
        q = self.q(queries).reshape(B, Q, self.heads, self.head_dim).transpose(1, 2)            # [B, h, Q, d]
        k, v = self.kv(context).reshape(B, L, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        r = attend(q, k, v, key_mask, self.impl, self.attn_drop if self.training else 0.0, return_attn)
        out, attn = r if return_attn else (r, None)
        out = self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, Q, E)))
        return (out, attn) if return_attn else out
