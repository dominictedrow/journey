"""Study-level heads: a bag of window vectors [B, L, F] (+ mask) → 12 logits [B, 12] (02 §C5, §C9).

Shared by both tracks (Track A feeds CoAtNet window features, Track B feeds SliceViT window vectors).
Every head ends in 12 independent logits (sigmoid + BCE; multi-label, not softmax), and every final logit layer is
**zero-initialized**, so at init every logit is 0, every probability 0.5, and the BCE is ln 2 ≈ 0.693 per label.

- `AttnMILHead`: per-finding attention MIL, the public "Raptor" head [KT §1.1]: for each finding k, a softmax over the
  windows says which windows matter for k; the finding's feature is that weighted average; one linear per finding.
  gated=False is Raptor exactly (Linear → Tanh → Dropout → Linear); gated=True adds the sigmoid gate of Ilse et al. 2018
  (the Global96 variant).
- `FindingQueryHead`: the transformer form of the same idea. A shallow self-attention stack lets windows see each
  other (across slices and series), then 12 learned query tokens ("12 questions") cross-attend over the windows.
- `ClsStudyHead`: the barebones plan: a [CLS] token + a transformer over the windows, then Linear(E, 12). Ablation.
"""
from __future__ import annotations

import torch
from torch import nn

from knee.model.attention import _safe_key_mask
from knee.model.block import Block, CrossBlock
from knee.model.slice_vit import init_weights

N_LABELS = 12


def _masked_softmax(scores: torch.Tensor, key_mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    """An empty bag (no real window) would give softmax(all -inf) = NaN; `_safe_key_mask` unmasks key 0 there, the
    same guard as the attention heads, and the caller zeroes that bag's pooled features."""
    if key_mask is not None:
        scores = scores.masked_fill(~_safe_key_mask(key_mask)[..., None], float("-inf"))
    return scores.float().softmax(dim=dim)


class AttnMILHead(nn.Module):
    def __init__(self, in_dim: int, n_labels: int = N_LABELS, att_dim: int = 256, dropout: float = 0.2,
                 gated: bool = False, norm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim) if norm else nn.Identity()
        self.att_v = nn.Linear(in_dim, att_dim)
        self.att_u = nn.Linear(in_dim, att_dim) if gated else None
        self.att_drop = nn.Dropout(dropout)
        self.att_w = nn.Linear(att_dim, n_labels)
        self.cls_w = nn.Parameter(torch.zeros(n_labels, in_dim))  # one classifier vector per finding, zero-init
        self.cls_b = nn.Parameter(torch.zeros(n_labels))
        self.apply(init_weights)

    def forward(self, tokens: torch.Tensor, key_mask: torch.Tensor | None = None, return_attn: bool = False):
        h = self.norm(tokens)                                                  # [B, L, F]
        a = torch.tanh(self.att_v(h))
        if self.att_u is not None:
            a = a * torch.sigmoid(self.att_u(h))
        attn = _masked_softmax(self.att_w(self.att_drop(a)), key_mask, dim=1)   # [B, L, 12], sums to 1 over L
        pooled = torch.einsum("blk,blf->bkf", attn.to(h.dtype), h)             # [B, 12, F]
        if key_mask is not None:                                               # empty bag → logits = bias only
            pooled = pooled * key_mask.any(dim=1)[:, None, None].to(pooled.dtype)
        logits = (pooled * self.cls_w).sum(-1) + self.cls_b                    # [B, 12]
        return (logits, attn.transpose(1, 2)) if return_attn else logits      # attn as [B, 12, L]


class FindingQueryHead(nn.Module):
    def __init__(self, in_dim: int, dim: int | None = None, n_labels: int = N_LABELS, ctx_depth: int = 2,
                 query_depth: int = 2, heads: int = 6, mlp_ratio: float = 4.0, qk_norm: bool = True,
                 drop_path: float = 0.0, layerscale: float | None = 0.1, attn_impl: str = "sdpa"):
        super().__init__()
        dim = dim or in_dim
        self.in_proj = nn.Linear(in_dim, dim) if dim != in_dim else nn.Identity()
        self.ctx_blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, qk_norm=qk_norm, drop_path=drop_path,
                                               layerscale=layerscale, attn_impl=attn_impl) for _ in range(ctx_depth)])
        self.queries = nn.Parameter(torch.zeros(1, n_labels, dim))
        self.query_blocks = nn.ModuleList([CrossBlock(dim, heads, mlp_ratio, qk_norm=qk_norm, drop_path=drop_path,
                                                      layerscale=layerscale, attn_impl=attn_impl)
                                           for _ in range(query_depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.cls_w = nn.Parameter(torch.zeros(n_labels, dim))  # logit_k = <q_k, w_k> + b_k
        self.cls_b = nn.Parameter(torch.zeros(n_labels))
        self.apply(init_weights)
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, tokens: torch.Tensor, key_mask: torch.Tensor | None = None, return_attn: bool = False):
        ctx = self.in_proj(tokens)
        for blk in self.ctx_blocks:
            ctx = blk(ctx, key_mask)
        q = self.queries.expand(tokens.shape[0], -1, -1).to(ctx.dtype)
        attn = None
        for i, blk in enumerate(self.query_blocks):
            last = return_attn and i == len(self.query_blocks) - 1
            r = blk(q, ctx, key_mask, return_attn=last)
            q, attn = r if last else (r, attn)
        q = self.norm(q)                                                       # [B, 12, E]
        logits = (q * self.cls_w).sum(-1) + self.cls_b
        return (logits, attn.mean(1)) if return_attn else logits              # attn: head-mean [B, 12, L]


class ClsStudyHead(nn.Module):
    def __init__(self, in_dim: int, dim: int | None = None, n_labels: int = N_LABELS, depth: int = 2, heads: int = 6,
                 mlp_ratio: float = 4.0, qk_norm: bool = True, drop_path: float = 0.0,
                 layerscale: float | None = 0.1, attn_impl: str = "sdpa"):
        super().__init__()
        dim = dim or in_dim
        self.in_proj = nn.Linear(in_dim, dim) if dim != in_dim else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, qk_norm=qk_norm, drop_path=drop_path,
                                           layerscale=layerscale, attn_impl=attn_impl) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc = nn.Linear(dim, n_labels)
        self.apply(init_weights)
        nn.init.zeros_(self.fc.weight)
        nn.init.trunc_normal_(self.cls_token, std=1e-6)

    def forward(self, tokens: torch.Tensor, key_mask: torch.Tensor | None = None, return_attn: bool = False):
        t = self.in_proj(tokens)
        t = torch.cat([self.cls_token.expand(t.shape[0], -1, -1).to(t.dtype), t], dim=1)
        if key_mask is not None:
            key_mask = torch.cat([key_mask.new_ones(key_mask.shape[0], 1), key_mask], dim=1)
        for blk in self.blocks:
            t = blk(t, key_mask)
        logits = self.fc(self.norm(t[:, 0]))
        return (logits, None) if return_attn else logits


def build_head(kind: str, in_dim: int, **kw) -> nn.Module:
    heads = {"attn_mil": AttnMILHead, "finding_query": FindingQueryHead, "cls": ClsStudyHead}
    if kind not in heads:
        raise ValueError(f"unknown head {kind!r}; choose from {sorted(heads)}")
    return heads[kind](in_dim, **kw)
