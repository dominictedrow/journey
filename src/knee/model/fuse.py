"""Inference-time weight folding (docs/plan/03_performance_plan.md, B2). Same outputs up to float rounding, fewer
memory passes. Only for eval: after folding, the modules no longer train the same way (the folded parameters are
gone), so call this on a loaded checkpoint, never on a model that is still training.

- LayerScale: `x + γ ⊙ Linear(h)` = `x + Linear'(h)` with W' = diag(γ)·W, b' = γ ⊙ b. Folded into `attn.proj` and
  `mlp.fc2` of every Block / CrossBlock (Dropout and DropPath are identities in eval), so one elementwise pass over
  the token tensor disappears per branch.
- BatchNorm after a conv (Track A's CoAtNet): in eval BN is a fixed per-channel affine, y = a·conv(x) + c with
  a = γ/√(var+ε), c = β − a·mean, so it folds into the conv: W' = a·W, b' = a·b + c. timm's BatchNormAct2d is BN →
  (drop) → act; the BN part folds and only the activation stays. BNs that come *before* a conv with an activation in
  between (CoAtNet's `pre_norm`) can't fold and are left alone.
"""
from __future__ import annotations

import torch
from torch import nn

from knee.model.block import Block, CrossBlock


@torch.no_grad()
def fold_layerscale(model: nn.Module) -> int:
    """Fold every LayerScale of our transformer blocks into the Linear before it. Returns the number folded."""
    n = 0
    for m in model.modules():
        if not isinstance(m, (Block, CrossBlock)):
            continue
        for ls, lin in ((m.ls1, m.attn.proj), (m.ls2, m.mlp.fc2)):
            if ls.gamma is None:
                continue
            g = ls.gamma.to(lin.weight.dtype)
            lin.weight.mul_(g[:, None])
            if lin.bias is not None:
                lin.bias.mul_(g)
            ls.gamma = None  # LayerScale.forward is now the identity
            n += 1
    return n


@torch.no_grad()
def fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BatchNorm without running statistics can't be folded")
    a = bn.running_var.add(bn.eps).rsqrt()
    if bn.weight is not None:
        a = a * bn.weight
    c = -bn.running_mean * a
    if bn.bias is not None:
        c = c + bn.bias
    conv.weight.mul_(a.reshape(-1, 1, 1, 1).to(conv.weight.dtype))
    b = c if conv.bias is None else conv.bias * a + c
    conv.bias = nn.Parameter(b.to(conv.weight.dtype))


class _ActOnly(nn.Module):
    """What is left of timm's BatchNormAct2d after its BN part was folded into the conv before it."""

    def __init__(self, drop: nn.Module, act: nn.Module):
        super().__init__()
        self.drop, self.act = drop, act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.drop(x))


def _is_bn_act(m: nn.Module) -> bool:
    return isinstance(m, nn.BatchNorm2d) and hasattr(m, "act") and hasattr(m, "drop")  # timm BatchNormAct2d


# (conv attribute, norm attribute) pairs where the norm directly follows the conv in timm's MaxxVit/CoAtNet forward:
# Stem: conv1 → norm1; MbConvBlock: conv1_1x1 → norm1, conv2_kxk → (se_early) → norm2.
_CONV_NORM_PAIRS = (("conv1", "norm1"), ("conv1_1x1", "norm1"), ("conv2_kxk", "norm2"))


@torch.no_grad()
def fold_timm_conv_bn(model: nn.Module) -> int:
    """Fold the BatchNormAct2d layers that directly follow a conv in timm CoAtNet/MaxxVit stems and MBConv blocks.
    Returns the number folded. Only these known module layouts are touched; anything else is left as is."""
    n = 0
    for m in model.modules():
        cls = type(m).__name__
        if cls not in ("Stem", "MbConvBlock"):
            continue
        for conv_name, norm_name in _CONV_NORM_PAIRS:
            conv, norm = getattr(m, conv_name, None), getattr(m, norm_name, None)
            if not isinstance(conv, nn.Conv2d) or norm is None or not _is_bn_act(norm):
                continue
            if cls == "Stem" and conv_name != "conv1":
                continue
            if conv_name == "conv2_kxk" and getattr(m, "se_early", None) is not None:
                continue  # SE sits between the conv and the norm here
            fold_bn_into_conv(conv, norm)
            setattr(m, norm_name, _ActOnly(norm.drop, norm.act))
            n += 1
    return n


def prepare_for_inference(model: nn.Module) -> dict[str, int]:
    """eval() + every exact fold that applies to this model. Returns what was folded (for the run log)."""
    model.eval()
    return {"layerscale": fold_layerscale(model), "conv_bn": fold_timm_conv_bn(model)}
