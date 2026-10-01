"""AdamW parameter groups (no weight decay on norms/biases/positions/tokens; optional layer-wise LR decay) and the
warmup + cosine schedule (01 §2 #13, 02 §C15).

LLRD (layer-wise LR decay, for fine-tuning a pretrained encoder): layer i of n gets lr · decay^(n-1-i). The top (head,
aggregator) trains at the full LR; the tokenizer, which holds the most generic features, moves least.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn

# slice_pos.proj maps slice positions to an embedding: a position table in all but form, so no decay. The heads'
# cls_w stays decayed on purpose: it is the classifier weight (a Linear in all but form), which ViT recipes decay.
NO_DECAY_NAMES = ("cls_token", "queries", "mask_token", "pos2d.table", "series_type.tables", "slot_embed.tables",
                  "slice_pos.proj.weight")


def no_weight_decay(name: str, p: torch.Tensor) -> bool:
    return p.ndim <= 1 or any(k in name for k in NO_DECAY_NAMES)


def param_groups(model: nn.Module, lr: float, weight_decay: float, layer_decay: float | None = None,
                 layer_id: Callable[[str], int] | None = None, n_layers: int | None = None,
                 lr_mult: Callable[[str], float] | None = None) -> list[dict]:
    """Groups keyed by (lr scale, decay/no-decay). `lr_mult(name)` can scale a subset (e.g. Track A backbone)."""
    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        scale = 1.0
        if layer_decay is not None and layer_decay != 1.0:
            scale *= layer_decay ** (n_layers - 1 - layer_id(name))
        if lr_mult is not None:
            scale *= lr_mult(name)
        wd = 0.0 if no_weight_decay(name, p) else weight_decay
        key = (round(scale, 8), wd)
        g = groups.setdefault(key, {"params": [], "lr": lr * scale, "weight_decay": wd, "lr_scale": scale,
                                    "names": []})
        g["params"].append(p)
        g["names"].append(name)
    return list(groups.values())


def build_adamw(groups: list[dict], betas=(0.9, 0.999), fused: bool | None = None) -> torch.optim.AdamW:
    clean = [{k: v for k, v in g.items() if k != "names"} for g in groups]
    kw = {"betas": tuple(betas)}
    if fused is not None:
        kw["fused"] = fused
    return torch.optim.AdamW(clean, **kw)


def warmup_cosine(step: int, warmup: int, total: int, min_ratio: float = 0.0) -> float:
    """Multiplier in [min_ratio, 1]: linear 0→1 over `warmup` steps, then a cosine down to min_ratio at `total`."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def set_lr(opt: torch.optim.Optimizer, base_lr: float, mult: float) -> float:
    """Set every group's LR to base_lr · its lr_scale · mult (explicit, so a resumed run gets the exact same LRs)."""
    for g in opt.param_groups:
        g["lr"] = base_lr * g.get("lr_scale", 1.0) * mult
    return base_lr * mult
