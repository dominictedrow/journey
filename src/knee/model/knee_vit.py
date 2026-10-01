"""KneeViT: the whole Track B model. A study = a variable-length bag of 2.5D windows from every series (02 §C).

    batch (from dataset.collate_bag):
      x            uint8 [B, L, 3, H, W]   the windows, padded to the longest bag
      window_mask  bool  [B, L]            True = real window
      z, z_mm      float [B, L]            slice position in the stack (0..1) and mm from the stack middle
      plane, fluid, contrast, field, slot, series_idx   long [B, L]   the window's series type

    1. real windows only → SliceViT (in chunks)        → [n_real, E]  scattered back to [B, L, E]
    2. + SlicePosEmbed(z [, z_mm]) + SeriesTypeEmbed   → [B, L, E]
    3. study head over the bag (key-padding masked)    → [B, 12] logits

Padding never reaches SliceViT (no wasted compute) and is masked in the head, so padded windows cannot change the
output (tests/test_model.py checks this). Training-time **window dropout** drops random real windows before step 1
(augmentation + fewer windows to encode).
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from knee.model.compiled import compile_module, default_buckets, encode_chunks
from knee.model.heads import build_head
from knee.model.pos_embed import SeriesTypeEmbed, SlicePosEmbed
from knee.model.slice_vit import SliceViT, to_input

TYPE_KEYS = ("plane", "fluid", "contrast", "field", "slot", "series_idx")

DEFAULT_CONFIG: dict = {
    "slice_vit": {"in_ch": 3, "img_size": 256, "patch": 16, "dim": 384, "depth": 12, "heads": 6, "mlp_ratio": 4.0,
                  "tokenizer": "conv_stem", "stem_norm": "ln", "pos2d": "sincos", "pool": "gap", "qk_norm": True,
                  "layerscale": 0.1, "drop_path": 0.1, "attn_impl": "sdpa", "grad_checkpointing": False},
    "slice_pos": {"n_freq": 8, "use_mm": False},
    "series_type": {"fields": ["plane", "fluid", "contrast", "field"], "max_series": 0},
    "head": {"kind": "attn_mil"},
    "window_chunk": 128,     # windows per SliceViT call (memory knob only; the output does not depend on it)
    "window_dropout": 0.0,   # HP: fraction of real windows dropped per study in training
}


def merge_config(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge_config(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def drop_windows(mask: torch.Tensor, p: float) -> torch.Tensor:
    """Drop a random fraction p of each study's real windows (training augmentation); every study keeps one."""
    if p <= 0:
        return mask
    keep = (torch.rand(mask.shape, device=mask.device) >= p) & mask
    none_left = ~keep.any(dim=1) & mask.any(dim=1)
    first = torch.zeros_like(mask).scatter_(1, mask.float().argmax(dim=1, keepdim=True), True)
    return keep | (first & none_left[:, None])  # branch-free: no GPU→CPU sync


class KneeViT(nn.Module):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg = merge_config(DEFAULT_CONFIG, cfg)
        self.encoder = SliceViT(**cfg["slice_vit"])
        dim = self.encoder.dim
        self.slice_pos = SlicePosEmbed(dim, **cfg["slice_pos"])
        st = cfg["series_type"]
        self.series_type = SeriesTypeEmbed(dim, tuple(st["fields"]), st.get("max_series", 0))
        head_kw = {k: v for k, v in cfg["head"].items() if k != "kind"}
        self.head = build_head(cfg["head"]["kind"], dim, **head_kw)
        self.window_chunk = int(cfg["window_chunk"])
        self.window_dropout = float(cfg["window_dropout"])
        self.chunk_buckets: list[int] | None = None  # set by compile_encoder: pad ragged chunks to fixed shapes

    def compile_encoder(self, mode: str | None = None, buckets: list[int] | None = None, min_bucket: int = 4) -> None:
        """torch.compile the SliceViT in place, fed only fixed chunk shapes (knee.model.compiled). The output doesn't
        change: SliceViT treats windows independently, so padding a chunk can't affect the real windows."""
        self.chunk_buckets = sorted(buckets) if buckets else default_buckets(self.window_chunk, min_bucket)
        compile_module(self.encoder, mode)

    def encode_windows(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 or float [n, C, H, W] → [n, E], in chunks of window_chunk."""
        dt = self.encoder.norm.weight.dtype  # fp32 normally (autocast casts inside); fp16 after to_half
        outs = encode_chunks(self.encoder, (to_input(x) if x.dtype == torch.uint8 else x).to(dt), self.window_chunk,
                             self.chunk_buckets)
        return torch.cat(outs) if outs else x.new_zeros((0, self.encoder.dim), dtype=dt)

    def forward(self, batch: dict, return_attn: bool = False):
        x, mask = batch["x"], drop_windows(batch["window_mask"], self.window_dropout if self.training else 0.0)
        B, L = mask.shape
        feats = self.encode_windows(x[mask])                                   # [n_real, E]
        tokens = feats.new_zeros((B, L, feats.shape[-1]))
        tokens[mask] = feats
        tokens = self.slice_pos(tokens, batch["z"], batch.get("z_mm"))
        tokens = self.series_type(tokens, {k: batch[k] for k in TYPE_KEYS if k in batch})
        return self.head(tokens, mask, return_attn=return_attn)

    # ---- LLRD support: encoder layers get decayed LRs; everything after the encoder is "the top"
    def layer_id(self, name: str) -> int:
        if name.startswith("encoder."):
            return self.encoder.layer_id(name[len("encoder."):])
        return self.encoder.depth + 1

    @property
    def n_layers(self) -> int:
        return self.encoder.depth + 2

    def load_encoder(self, state: dict, strict: bool = True) -> tuple[list[str], list[str]]:
        """Load MAE-pretrained SliceViT weights (keys relative to SliceViT). Returns (missing, unexpected)."""
        r = self.encoder.load_state_dict(state, strict=strict)
        return list(r.missing_keys), list(r.unexpected_keys)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
