"""Losses for 12 independent findings on soft report labels (02 §C9).

- `soft_bce`: BCE against soft targets p ∈ [0,1], each (study, finding) entry weighted by its label confidence
  (S04's w_<label>), missing labels masked out. The weighted sum is divided by the number of labelled entries, not
  by the weight sum: with 1-study micro-batches a weight-sum denominator cancels the weights (an all-low-confidence
  study would train as hard as a confident one; review 2026-09-30).
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
    n = torch.ones_like(targets) if label_mask is None else label_mask.float()
    return (loss * w).sum() / n.sum().clamp(min=1)


def pairwise_rank_loss(logits: torch.Tensor, targets: torch.Tensor, label_mask: torch.Tensor | None = None,
                       ref_logits: torch.Tensor | None = None, ref_targets: torch.Tensor | None = None,
                       ref_mask: torch.Tensor | None = None, pos_thr: float = 0.6, neg_thr: float = 0.4
                       ) -> torch.Tensor:
    """Pairs within the batch, plus (current × reference) pairs if a reference set is given. Returns 0 (with grad)
    when no finding has a pair.

    Written with pair masks instead of boolean indexing: `s[pos, k]` has a data-dependent size, so every such index
    forces a GPU→CPU sync (36 per micro-batch with the bank). Here every (i, j, finding) pair is scored and the
    invalid ones are zeroed, which is the same mean per finding, then the mean over findings that have a pair."""
    s, y = logits.float(), targets.float()
    m = torch.ones_like(y, dtype=torch.bool) if label_mask is None else label_mask.bool()
    pos, neg = m & (y > pos_thr), m & (y < neg_thr)                                  # [B, K]
    terms = [(s[:, None, :] - s[None, :, :], pos[:, None, :] & neg[None, :, :])]    # [B, B, K]
    if ref_logits is not None and len(ref_logits):
        rs, ry = ref_logits.float().detach(), ref_targets.float()
        rm = torch.ones_like(ry, dtype=torch.bool) if ref_mask is None else ref_mask.bool()
        rpos, rneg = rm & (ry > pos_thr), rm & (ry < neg_thr)
        terms += [(s[:, None, :] - rs[None, :, :], pos[:, None, :] & rneg[None, :, :]),   # [B, R, K]
                  (rs[:, None, :] - s[None, :, :], rpos[:, None, :] & neg[None, :, :])]   # [R, B, K]
    # where, not multiply: 0 · NaN = NaN, so one non-finite logit would poison every finding's mean
    num = sum(torch.where(w, F.softplus(-d), 0.0).sum(dim=(0, 1)) for d, w in terms)   # [K]
    cnt = sum(w.sum(dim=(0, 1)) for _, w in terms)                                     # [K]
    has = cnt > 0
    per_label = torch.where(has, num / cnt.clamp(min=1), num * 0.0)
    return per_label.sum() / has.sum().clamp(min=1)


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
        """Rows with a non-finite logit (an fp16 overflow) are stored fully masked with zero logits, branch-free: a
        NaN in the bank would otherwise make the rank loss NaN for the next `size` studies, the scaler would skip
        every one of those steps, and each skip halves the loss scale."""
        l, t, m = logits.detach().float(), targets.detach().float(), mask.detach().bool()
        ok = torch.isfinite(l).all(dim=1, keepdim=True)
        l, m = torch.where(ok, l, 0.0), m & ok
        if self.logits is None:
            self.logits, self.targets, self.mask = l, t, m
        else:
            self.logits = torch.cat([self.logits, l])[-self.size:]
            self.targets = torch.cat([self.targets, t])[-self.size:]
            self.mask = torch.cat([self.mask, m])[-self.size:]
