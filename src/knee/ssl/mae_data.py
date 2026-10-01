"""MAE data: single 2.5D windows drawn from every series of the train cache (S03 brief, 02 §D).

One epoch = `windows_per_series` random windows from **each series** (weighting per series, not per slice: a 160-slice
3D GRE would otherwise outweigh five 30-slice series). Each window: a centre in the 2–98% span → the 3-slice window at
the cache's 384 px → a random resized crop (area fraction `scale`, aspect `ratio`) → resized to `out_px`.

Each item is **one read of one series that yields `windows_per_read` windows** (docs/plan/03_performance_plan.md, A3):
the file access, rng setup and Python overhead are paid once per read instead of once per window. With
windows_per_read = 1 an item is exactly the old single window (a uniform centre). With more, the centres are drawn by
stratified jitter over the span (one per equal bin, as in supervised training), so one read covers the stack instead
of possibly repeating a slice. Items are (windows_per_read, 3, out_px, out_px); batch them with `collate_windows`.

**gpu_crop** (docs/plan/03_performance_plan.md, A4): the crop + resize was the loader's CPU bottleneck (one cv2 call
per window, hundreds per second on 4 vCPUs). With gpu_crop=True an item is the raw windows at the cache's resolution
plus their crop boxes (drawn with exactly the same rng calls), and `gpu_resized_crop` does every crop and resize of a
batch in one `torchvision.ops.roi_align` call on the GPU. Workers then only read and copy. roi_align with adaptive
sampling averages ceil(crop/out)² bilinear samples per output pixel, so shrinking is anti-aliased like INTER_AREA
(close, not identical: [val]); enlarging is bilinear with half-pixel centres, like INTER_LINEAR. The output is rounded
to uint8, so the model sees the same input type either way. Cost: 3·384² bytes per window over PCIe (~57 MB per batch
of 128, a few ms).
Trade-off: a batch holds windows_per_read windows per series, i.e. fewer distinct series per batch (HP).
No flips (02 §C12: laterality is canonicalized; flips would teach nothing useful). No intensity jitter: the
per-patch-normalized MAE target ignores brightness anyway.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from knee.data.dataset import SPAN, CacheReader, _span_idx, gather_windows, train_centres


def crop_box(H: int, W: int, rng: np.random.Generator, scale=(0.6, 1.0), ratio=(0.85, 1.18)) -> tuple[int, ...]:
    """A random crop (y0, x0, h, w): area fraction ~ U(scale), aspect ratio log-uniform in `ratio`."""
    s = rng.uniform(*scale)
    r = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
    w = int(round(min(W, math.sqrt(s * r) * W)))
    h = int(round(min(H, math.sqrt(s / r) * H)))
    y0, x0 = int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))
    return y0, x0, h, w


def random_resized_crop(win: np.ndarray, out_px: int, rng: np.random.Generator, scale=(0.6, 1.0),
                        ratio=(0.85, 1.18)) -> np.ndarray:
    """win (C, S, S) uint8 → (C, out_px, out_px) uint8. scale = (1, 1) is a plain resize."""
    C, H, W = win.shape
    y0, x0, h, w = crop_box(H, W, rng, scale, ratio)
    crop = np.ascontiguousarray(win[:, y0:y0 + h, x0:x0 + w].transpose(1, 2, 0))  # HWC for cv2
    interp = cv2.INTER_AREA if h > out_px else cv2.INTER_LINEAR
    out = cv2.resize(crop, (out_px, out_px), interpolation=interp)
    return out.reshape(out_px, out_px, C).transpose(2, 0, 1)


def gpu_resized_crop(x: torch.Tensor, boxes: torch.Tensor, out_px: int) -> torch.Tensor:
    """x uint8 [N, C, S, S] (on the GPU), boxes float [N, 4] = (x0, y0, x1, y1) in pixels (crop = columns x0..x1-1)
    → uint8 [N, C, out_px, out_px]: every crop + resize of the batch in one roi_align call."""
    from torchvision.ops import roi_align
    rois = torch.cat([torch.arange(len(x), device=x.device, dtype=torch.float32)[:, None],
                      boxes.to(device=x.device, dtype=torch.float32)], dim=1)
    y = roi_align(x.float(), rois, (out_px, out_px), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
    return y.round_().clamp_(0, 255).to(torch.uint8)


def collate_windows(items: list):
    """Items → one batch: a uint8 tensor, or (uint8 windows, boxes) with gpu_crop."""
    if isinstance(items[0], torch.Tensor):
        return torch.cat(items)
    return torch.cat([i[0] for i in items]), torch.cat([i[1] for i in items])


class MAEWindowDataset(Dataset):
    def __init__(self, reader: CacheReader, study_uids: list[str] | None = None, out_px: int = 256,
                 windows_per_series: int = 1, span=SPAN, scale=(0.6, 1.0), ratio=(0.85, 1.18), seed: int = 0,
                 windows_per_read: int = 1, gpu_crop: bool = False):
        if windows_per_read < 1 or windows_per_series % windows_per_read:
            raise ValueError(f"windows_per_series ({windows_per_series}) must be a multiple of windows_per_read "
                             f"({windows_per_read})")
        self.r, self.px, self.k, self.span, self.scale, self.ratio, self.seed = \
            reader, out_px, windows_per_series, span, scale, ratio, seed
        self.kr, self.gpu_crop = windows_per_read, gpu_crop
        uids = reader.study_uids if study_uids is None else study_uids
        rows = [s for u in uids for s in reader.series(u) if int(s["n_slices"]) > 0]
        # compact index (cheap to pickle into every Windows spawn worker)
        self.shard = np.array([int(s["shard"]) for s in rows], dtype=np.int32)
        self.offset = np.array([int(s["offset"]) for s in rows], dtype=np.int64)
        self.n = np.array([int(s["n_slices"]) for s in rows], dtype=np.int32)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @property
    def n_series(self) -> int:
        return len(self.n)

    def __len__(self) -> int:
        """Items (series reads) per epoch."""
        return self.n_series * (self.k // self.kr)

    @property
    def n_windows(self) -> int:
        return len(self) * self.kr

    def __getitem__(self, i: int):
        """(windows_per_read, 3, out_px, out_px) uint8, deterministic given (seed, epoch, i): resumable.
        gpu_crop: (windows_per_read, 3, S, S) uint8 at the cache resolution + boxes (windows_per_read, 4)."""
        j = i % self.n_series
        rng = np.random.default_rng((self.seed, self.epoch, i))
        vol = self.r.pixels({"shard": self.shard[j], "offset": self.offset[j], "n_slices": self.n[j]})
        if self.kr == 1:
            lo, hi = _span_idx(int(self.n[j]), self.span)
            c = np.array([rng.integers(lo, hi + 1)])
        else:
            c = train_centres(int(self.n[j]), self.kr, rng, self.span)
        wins = gather_windows(vol, c)                                     # (kr, 3, 384, 384), one read
        if self.gpu_crop:
            S = wins.shape[-1]
            b = [crop_box(S, S, rng, self.scale, self.ratio) for _ in wins]   # same rng calls as the CPU path
            boxes = torch.tensor([[x0, y0, x0 + w, y0 + h] for y0, x0, h, w in b], dtype=torch.float32)
            return torch.from_numpy(np.ascontiguousarray(wins)), boxes
        return torch.from_numpy(np.stack([random_resized_crop(w, self.px, rng, self.scale, self.ratio)
                                          for w in wins]))
