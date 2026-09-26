"""Losses for 12 independent findings on soft report labels (02 §C9).

- `soft_bce`: BCE against soft targets p ∈ [0,1], each (study, finding) entry weighted by its label confidence
  (S04's w_<label>), missing labels masked out.
- `pairwise_rank_loss`: the metric is AUC, i.e. "does a positive study score above a negative one". For each finding,
  softplus(-(s_pos - s_neg)) over (positive, negative) pairs, averaged, then averaged over findings (macro, like the
  metric). Public stacks use it at weight ≈ 0.05–0.12 [KT].
- `RankBank`: our micro-batches are 1–2 studies (12 GB VRAM), so a batch rarely contains a positive-negative pair.
  The bank keeps the detached logits + targets of the last `size` studies; the current studies are ranked against
  them (gradient flows through the current logits only). [derived] This approximates the large-batch rank loss;
  the banked logits are slightly stale (a few hundred steps at most). HP: size.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_bce(logits: torch.Tensor, targets: torch.Tensor, label_mask: torch.Tensor | None = None,
             weights: torch.Tensor | None = None) -> torch.Tensor:
    w = torch.ones_like(targets) if weights is None else weights.float()
    if label_mask is not None:
        w = w * label_mask.float()
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    return (loss * w).sum() / w.sum().clamp(min=1e-6)


def pairwise_rank_loss(logits: torch.Tensor, targets: torch.Tensor, label_mask: torch.Tensor | None = None,
                       ref_logits: torch.Tensor | None = None, ref_targets: torch.Tensor | None = None,
                       ref_mask: torch.Tensor | None = None, pos_thr: float = 0.6, neg_thr: float = 0.4
                       ) -> torch.Tensor:
    """Pairs within the batch, plus (current × reference) pairs if a reference set is given. Returns 0 (with grad)
    when no finding has a pair."""
    s, y = logits.float(), targets.float()
    m = torch.ones_like(y, dtype=torch.bool) if label_mask is None else label_mask.bool()
    if ref_logits is not None and len(ref_logits):
        rs, ry = ref_logits.float().detach(), ref_targets.float()
        rm = torch.ones_like(ry, dtype=torch.bool) if ref_mask is None else ref_mask.bool()
    else:
        rs = ry = rm = None
    per_label = []
    for k in range(y.shape[1]):
        pos, neg = m[:, k] & (y[:, k] > pos_thr), m[:, k] & (y[:, k] < neg_thr)
        diffs = [s[pos, k][:, None] - s[neg, k][None, :]]
        if rs is not None:
            rpos, rneg = rm[:, k] & (ry[:, k] > pos_thr), rm[:, k] & (ry[:, k] < neg_thr)
            diffs += [s[pos, k][:, None] - rs[rneg, k][None, :], rs[rpos, k][:, None] - s[neg, k][None, :]]
        d = torch.cat([x.flatten() for x in diffs])
        if d.numel():
            per_label.append(F.softplus(-d).mean())
    if not per_label:
        return logits.sum() * 0.0
    return torch.stack(per_label).mean()


class RankBank:
    def __init__(self, size: int = 256):
        self.size = size
        self.logits: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None

    def get(self):
        return self.logits, self.targets, self.mask

    @torch.no_grad()
    def push(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> None:
        l, t, m = logits.detach().float(), targets.detach().float(), mask.detach().bool()
        if self.logits is None:
            self.logits, self.targets, self.mask = l, t, m
        else:
            self.logits = torch.cat([self.logits, l])[-self.size:]
            self.targets = torch.cat([self.targets, t])[-self.size:]
            self.mask = torch.cat([self.mask, m])[-self.size:]
