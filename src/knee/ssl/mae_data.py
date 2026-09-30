"""MAE data: single 2.5D windows drawn from every series of the train cache (S03 brief, 02 §D).

One epoch = `windows_per_series` random windows from **each series** (weighting per series, not per slice: a 160-slice
3D GRE would otherwise outweigh five 30-slice series). Each window: a centre in the 2–98% span → the 3-slice window at
the cache's 384 px → a random resized crop (area fraction `scale`, aspect `ratio`) → resized to `out_px`.

Each item is **one read of one series that yields `windows_per_read` windows** (docs/plan/03_performance_plan.md, A3):
the file access, rng setup and Python overhead are paid once per read instead of once per window. With
windows_per_read = 1 an item is exactly the old single window (a uniform centre). With more, the centres are drawn by
stratified jitter over the span (one per equal bin, as in supervised training), so one read covers the stack instead
of possibly repeating a slice. Items are (windows_per_read, 3, out_px, out_px); batch them with collate_fn=torch.cat.
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


def random_resized_crop(win: np.ndarray, out_px: int, rng: np.random.Generator, scale=(0.6, 1.0),
                        ratio=(0.85, 1.18)) -> np.ndarray:
    """win (C, S, S) uint8 → (C, out_px, out_px) uint8. scale = (1, 1) is a plain resize."""
    C, H, W = win.shape
    s = rng.uniform(*scale)
    r = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
    w = int(round(min(W, math.sqrt(s * r) * W)))
    h = int(round(min(H, math.sqrt(s / r) * H)))
    y0, x0 = int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))
    crop = np.ascontiguousarray(win[:, y0:y0 + h, x0:x0 + w].transpose(1, 2, 0))  # HWC for cv2
    interp = cv2.INTER_AREA if h > out_px else cv2.INTER_LINEAR
    out = cv2.resize(crop, (out_px, out_px), interpolation=interp)
    return out.reshape(out_px, out_px, C).transpose(2, 0, 1)


class MAEWindowDataset(Dataset):
    def __init__(self, reader: CacheReader, study_uids: list[str] | None = None, out_px: int = 256,
                 windows_per_series: int = 1, span=SPAN, scale=(0.6, 1.0), ratio=(0.85, 1.18), seed: int = 0,
                 windows_per_read: int = 1):
        if windows_per_read < 1 or windows_per_series % windows_per_read:
            raise ValueError(f"windows_per_series ({windows_per_series}) must be a multiple of windows_per_read "
                             f"({windows_per_read})")
        self.r, self.px, self.k, self.span, self.scale, self.ratio, self.seed = \
            reader, out_px, windows_per_series, span, scale, ratio, seed
        self.kr = windows_per_read
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

    def __getitem__(self, i: int) -> torch.Tensor:
        """(windows_per_read, 3, out_px, out_px) uint8, deterministic given (seed, epoch, i): resumable."""
        j = i % self.n_series
        rng = np.random.default_rng((self.seed, self.epoch, i))
        vol = self.r.pixels({"shard": self.shard[j], "offset": self.offset[j], "n_slices": self.n[j]})
        if self.kr == 1:
            lo, hi = _span_idx(int(self.n[j]), self.span)
            c = np.array([rng.integers(lo, hi + 1)])
        else:
            c = train_centres(int(self.n[j]), self.kr, rng, self.span)
        wins = gather_windows(vol, c)                                     # (kr, 3, 384, 384), one read
        return torch.from_numpy(np.stack([random_resized_crop(w, self.px, rng, self.scale, self.ratio)
                                          for w in wins]))
