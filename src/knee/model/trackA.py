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

from knee.model.compiled import compile_module, default_buckets, encode_chunks
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
               "slice_pos": False, "freeze_bn": True, **(cfg or {})}
        self.cfg = cfg
        self.backbone = timm.create_model(cfg["backbone"], pretrained=cfg["pretrained"], num_classes=0)
        self.set_grad_checkpointing(cfg["grad_checkpointing"])
        self.freeze_bn = bool(cfg["freeze_bn"])
        if self.freeze_bn:
            for m in self.bn_modules():
                for prm in m.parameters(recurse=False):
                    prm.requires_grad_(False)
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
        self.chunk_buckets: list[int] | None = None  # set by compile_encoder
        self.channels_last = False

    def set_grad_checkpointing(self, which) -> None:
        """True/False = every stage (timm's switch), or a list of stage indices to checkpoint only those.

        Selective is the cheaper trade. Per window of coatnet_rmlp_2 at 384 px (measured 2026-10-01 with
        saved-tensor hooks + the flop counter on CPU; scripts/perf/stage_memory.py): stages 0–1 (MBConv at 96² and
        48²) keep ~43% of the activations but cost ~24% of the forward FLOPs; stage 2 (transformer, 24²) keeps ~52%
        for ~63%. So `[0, 1]` recomputes about a quarter of the forward (+~8% step time, UNVERIFIED on a GPU) and
        frees roughly 40% of the activation memory, versus +30% for checkpointing everything (T4, §2b)."""
        if isinstance(which, bool) or which is None:
            self.backbone.set_grad_checkpointing(bool(which))
            return
        stages = list(self.backbone.stages)
        bad = [i for i in which if not 0 <= int(i) < len(stages)]
        if bad:
            raise ValueError(f"grad_checkpointing stages {bad} out of range (backbone has {len(stages)} stages)")
        self.backbone.set_grad_checkpointing(False)
        for i in which:
            stages[int(i)].grad_checkpointing = True

    def bn_modules(self) -> list[nn.Module]:
        """The backbone's BatchNorm layers (timm's BatchNormAct2d subclasses nn.BatchNorm2d)."""
        return [m for m in self.backbone.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def train(self, mode: bool = True):
        """freeze_bn (default): BatchNorm stays in eval mode (pretrained running statistics, frozen affine) while the
        rest trains. A batch here is the windows of one or two knees, highly correlated, so batch statistics are noisy
        and the running averages built from them don't match what eval sees; top-k epoch averaging would also average
        running statistics that drift between epochs, and with timm's gradient checkpointing the recompute in
        backward updates them a second time (review 2026-09-30, bug.md #6)."""
        super().train(mode)
        if mode and self.freeze_bn:
            for m in self.bn_modules():
                m.eval()
        return self

    def use_channels_last(self) -> None:
        """NHWC weights and inputs: cuDNN's tensor-core layout (−5% train, −3% inference on the T4, §2b). Exact."""
        self.channels_last = True
        self.backbone.to(memory_format=torch.channels_last)

    def compile_encoder(self, mode: str | None = None, buckets: list[int] | None = None, min_bucket: int = 4) -> None:
        """torch.compile the backbone in place, fed only fixed chunk shapes. Exact in eval, and in training with
        freeze_bn (the default); with freeze_bn off, BatchNorm sees the padded copies in the ragged chunk's batch
        statistics (knee.model.compiled explains why that's small)."""
        self.chunk_buckets = sorted(buckets) if buckets else default_buckets(self.window_chunk, min_bucket)
        compile_module(self.backbone, mode)

    def _encode_chunk(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 chunk → normalized float → backbone. Casting per chunk keeps only one chunk in float at a time (the
        whole Track A eval bag in fp32 is ~0.4 GB; review 2026-09-30)."""
        dt = self.mean.dtype  # fp32 normally; fp16 after to_half
        x = ((x.float() / 255.0 - self.mean.float()) / self.std.float()).to(dt)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return self.backbone(x)

    def encode_windows(self, x: torch.Tensor) -> torch.Tensor:
        outs = encode_chunks(self._encode_chunk, x, self.window_chunk, self.chunk_buckets)
        return torch.cat(outs) if outs else x.new_zeros((0, self.dim), dtype=self.mean.dtype)

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
