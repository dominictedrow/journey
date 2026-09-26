"""Track A: a pretrained timm backbone per 2.5D window + the shared AttnMILHead (S05, 02 §E).

The only file in knee.model that imports timm (Track B must not use timm model classes; decision 2026-09-24).
Default backbone: coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k (the public Raptor recipe, 768-d GAP features, 384 px).

Input = the same bag format as KneeViT (dataset.collate_bag), or the 5-slot grid of StudyWindowsDataset turned into a
bag with `slots_to_bag`. Pixels are uint8 [0,255] → [0,1] → ImageNet mean/std (what the backbone was trained on;
the public CoAtNet arms do the same [KT §1.1]).
"""
from __future__ import annotations

import torch
from torch import nn

from knee.model.heads import build_head
from knee.model.knee_vit import drop_windows
from knee.model.pos_embed import SeriesTypeEmbed, SlicePosEmbed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def slots_to_bag(batch: dict) -> dict:
    """StudyWindowsDataset batch (x [B, S, W, 3, h, w], slot_mask [B, S], z [B, S, W]) → the bag format."""
    x = batch["x"]
    B, S, W = x.shape[:3]
    slot = torch.arange(S, device=x.device).repeat_interleave(W).expand(B, -1)
    return {"x": x.reshape(B, S * W, *x.shape[3:]),
            "window_mask": batch["slot_mask"][:, :, None].expand(B, S, W).reshape(B, S * W),
            "z": batch["z"].reshape(B, S * W), "slot": slot,
            **{k: batch[k] for k in ("labels", "label_mask", "study_uid") if k in batch}}


class TrackAModel(nn.Module):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        import timm  # local import: Track B code paths never load timm

        cfg = {"backbone": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "pretrained": True, "head": {"kind": "attn_mil"},
               "window_chunk": 32, "window_dropout": 0.0, "grad_checkpointing": True, "slot_embed": True,
               "slice_pos": False, **(cfg or {})}
        self.cfg = cfg
        self.backbone = timm.create_model(cfg["backbone"], pretrained=cfg["pretrained"], num_classes=0)
        if cfg["grad_checkpointing"]:
            self.backbone.set_grad_checkpointing(True)
        dim = self.backbone.num_features
        self.dim = dim
        # Optional extras on top of the public recipe (off = Raptor exactly). Zero/small init, so they start neutral.
        self.slot_embed = SeriesTypeEmbed(dim, ("slot",)) if cfg["slot_embed"] else None
        if self.slot_embed is not None:
            nn.init.zeros_(self.slot_embed.tables["slot"].weight)
        self.slice_pos = SlicePosEmbed(dim) if cfg["slice_pos"] else None
        head_kw = {k: v for k, v in cfg["head"].items() if k != "kind"}
        self.head = build_head(cfg["head"]["kind"], dim, **head_kw)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.window_chunk = int(cfg["window_chunk"])
        self.window_dropout = float(cfg["window_dropout"])

    def encode_windows(self, x: torch.Tensor) -> torch.Tensor:
        outs = [self.backbone((c.float() / 255.0 - self.mean) / self.std) for c in x.split(self.window_chunk)]
        return torch.cat(outs) if outs else x.new_zeros((0, self.dim), dtype=torch.float32)

    def forward(self, batch: dict, return_attn: bool = False):
        if batch["x"].ndim == 6:
            batch = slots_to_bag(batch)
        mask = drop_windows(batch["window_mask"], self.window_dropout if self.training else 0.0)
        B, L = mask.shape
        feats = self.encode_windows(batch["x"][mask])
        tokens = feats.new_zeros((B, L, feats.shape[-1]))
        tokens[mask] = feats
        if self.slot_embed is not None:
            tokens = self.slot_embed(tokens, {"slot": batch["slot"]})
        if self.slice_pos is not None:
            tokens = self.slice_pos(tokens, batch["z"])
        return self.head(tokens, mask, return_attn=return_attn)

    def param_groups_split(self) -> tuple[list, list]:
        """(backbone params, everything else): the backbone gets a much lower LR than the head (02 §E)."""
        bb = [p for n, p in self.named_parameters() if n.startswith("backbone.") and p.requires_grad]
        rest = [p for n, p in self.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
        return bb, rest
