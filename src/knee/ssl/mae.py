"""Masked autoencoder around SliceViT (He et al. 2022; 02 §D).

    window [N, 3, H, W]
      → patch tokens [N, n, E] (+ pos2d)
      → keep a random 25% per image (mask_ratio 0.75, HP)     ← the encoder only ever sees these: ~4× cheaper
      → SliceViT blocks → [N, n_keep, E]
      → decoder: Linear(E → D) + a shared [MASK] token in every hidden position, + its own sincos pos,
        a few narrow blocks, Linear(D → P·P·C) = predicted pixels of every patch
      → loss = MSE on the masked patches only, against per-patch normalized pixels (norm_pix: each target patch
        minus its mean, divided by its std, which makes the model predict structure, not brightness)

After pretraining, only the encoder (SliceViT) is kept; the decoder is thrown away (02 §C11).

Conv stem caveat [derived]: the 3×3 convs of ConvStem see a few pixels past their patch edge, so a visible token
could read the border of a masked neighbour. `zero_masked_pixels` (default on for the conv stem) blanks the masked
patches in pixel space before the tokenizer, so there is nothing to leak. Patchify doesn't need it (no overlap).
"""
from __future__ import annotations

import torch
from torch import nn

from knee.model.block import Block
from knee.model.patch_embed import ConvStem
from knee.model.pos_embed import sincos_2d
from knee.model.slice_vit import INPUT_MEAN, INPUT_STD, SliceViT, init_weights, to_input


def patchify(imgs: torch.Tensor, p: int) -> torch.Tensor:
    """[N, C, H, W] → [N, n, P·P·C], row-major patches (the same order as the tokenizers)."""
    N, C, H, W = imgs.shape
    x = imgs.reshape(N, C, H // p, p, W // p, p)
    return torch.einsum("nchpwq->nhwpqc", x).reshape(N, (H // p) * (W // p), p * p * C)


def unpatchify(x: torch.Tensor, p: int, grid: tuple[int, int], c: int) -> torch.Tensor:
    gh, gw = grid
    x = x.reshape(x.shape[0], gh, gw, p, p, c)
    return torch.einsum("nhwpqc->nchpwq", x).reshape(x.shape[0], c, gh * p, gw * p)


def random_masking(n_img: int, n_tok: int, ratio: float, device, generator=None):
    """Per-image random subset. Returns ids_keep [N, n_keep], ids_restore [N, n], mask [N, n] (1 = masked)."""
    n_keep = max(1, int(round(n_tok * (1 - ratio))))
    noise = torch.rand(n_img, n_tok, device=device, generator=generator)
    ids_shuffle = noise.argsort(dim=1)            # ascending: the first n_keep are kept
    ids_restore = ids_shuffle.argsort(dim=1)
    mask = torch.ones(n_img, n_tok, device=device)
    mask[:, :n_keep] = 0
    return ids_shuffle[:, :n_keep], ids_restore, torch.gather(mask, 1, ids_restore)


class MAE(nn.Module):
    def __init__(self, encoder: SliceViT, mask_ratio: float = 0.75, dec_dim: int = 256, dec_depth: int = 4,
                 dec_heads: int = 8, norm_pix: bool = True, zero_masked_pixels: bool | None = None,
                 in_ch: int = 3, norm_pix_eps: float = 1e-6):
        """norm_pix_eps: the variance floor of the per-patch target normalization, in model-input units (to_input:
        one grey level = 1/255/0.5 ≈ 0.008). At 1e-6 (the MAE paper's) a background patch with a grey level or two of
        noise (variance ~1e-4) is stretched to unit variance, so the decoder is asked to predict noise; ~1e-2 keeps
        such patches near 0. An ablation for the MAE pilot (review 2026-09-30); the default is unchanged."""
        super().__init__()
        self.encoder, self.mask_ratio, self.norm_pix, self.in_ch = encoder, mask_ratio, norm_pix, in_ch
        self.norm_pix_eps = float(norm_pix_eps)
        self.p = encoder.patch
        self.zero_masked_pixels = isinstance(encoder.tokenizer, ConvStem) if zero_masked_pixels is None \
            else zero_masked_pixels
        E = encoder.dim
        self.dec_embed = nn.Linear(E, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads, 4.0, qk_norm=True, layerscale=None)
                                         for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim, eps=1e-6)
        self.dec_pred = nn.Linear(dec_dim, self.p * self.p * in_ch)
        for m in (self.dec_embed, self.dec_blocks, self.dec_norm, self.dec_pred):
            m.apply(init_weights)
        nn.init.normal_(self.mask_token, std=0.02)

    def _pixel_mask(self, mask: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        gh, gw = grid
        m = mask.reshape(-1, 1, gh, gw)
        return m.repeat_interleave(self.p, 2).repeat_interleave(self.p, 3)  # [N, 1, H, W], 1 = masked

    def forward(self, x_u8: torch.Tensor, generator=None) -> dict:
        """x_u8 uint8 [N, C, H, W] (H, W multiples of P). Returns loss, pred [N, n, P·P·C], mask [N, n], target."""
        x = to_input(x_u8) if x_u8.dtype == torch.uint8 else x_u8
        N, _, H, W = x.shape
        grid = (H // self.p, W // self.p)
        n = grid[0] * grid[1]
        ids_keep, ids_restore, mask = random_masking(N, n, self.mask_ratio, x.device, generator)
        x_in = x * (1 - self._pixel_mask(mask, grid)) if self.zero_masked_pixels else x

        # encoder on the visible tokens only
        t, _ = self.encoder.tokenize(x_in)                                             # [N, n, E] with pos2d
        t = torch.gather(t, 1, ids_keep[..., None].expand(-1, -1, t.shape[-1]))        # [N, n_keep, E]
        t = self.encoder.run_blocks(self.encoder.add_cls(t))
        if self.encoder.cls_token is not None:
            t = t[:, 1:]

        # decoder over all positions
        d = self.dec_embed(t)
        d = torch.cat([d, self.mask_token.expand(N, n - d.shape[1], -1).to(d.dtype)], dim=1)
        d = torch.gather(d, 1, ids_restore[..., None].expand(-1, -1, d.shape[-1]))   # back to image order
        d = d + sincos_2d(*grid, d.shape[-1], d.device)[None].to(d.dtype)
        for blk in self.dec_blocks:
            d = blk(d)
        pred = self.dec_pred(self.dec_norm(d)).float()                                 # [N, n, P·P·C]

        target = patchify(x.float(), self.p)
        if self.norm_pix:
            mu, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
            target = (target - mu) / (var + self.norm_pix_eps).sqrt()
        per_patch = (pred - target).square().mean(-1)                                  # [N, n]
        loss = (per_patch * mask).sum() / mask.sum().clamp(min=1)
        return {"loss": loss, "pred": pred, "mask": mask, "target": target, "grid": grid}

    @torch.no_grad()
    def reconstruct(self, x_u8: torch.Tensor, generator=None) -> dict:
        """For the saved reconstruction images: input, masked input and the prediction in [0,1] pixel space
        (norm_pix predictions are re-scaled with the true patch mean/std, as in the MAE paper's figures)."""
        out = self.forward(x_u8, generator)
        x = x_u8.float() / 255.0
        tp = patchify(x, self.p)
        pred = out["pred"]
        if self.norm_pix:
            pred = pred * (tp.var(-1, keepdim=True) + self.norm_pix_eps).sqrt() + tp.mean(-1, keepdim=True)
        else:  # the target was the normalized model input
            pred = pred * INPUT_STD + INPUT_MEAN
        m = out["mask"][..., None]
        C = x.shape[1]
        return {"input": x, "masked": unpatchify(tp * (1 - m), self.p, out["grid"], C),
                "recon": unpatchify(pred.clamp(0, 1), self.p, out["grid"], C),
                "pasted": unpatchify(tp * (1 - m) + pred.clamp(0, 1) * m, self.p, out["grid"], C),
                "loss": out["loss"]}
