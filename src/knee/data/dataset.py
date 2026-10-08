"""PyTorch datasets over the S02 cache (S02 step 7). Shared by Track A and Track B; inference reuses the window
functions on in-memory arrays from preprocess_study, so train and test windows are assembled the same way.

A **2.5D window** = 3 physically adjacent slices of one series (centre ± 1) stacked as channels (decision 2026-09-24).
Train: K window centres per series drawn with stratified jitter over the span (a new view each epoch).
Eval:  evenly spaced centres over the span (deterministic), or every slice.

Datasets return **uint8** pixels (4× smaller to move between processes); convert to float on the GPU.
They are spawn-safe for Windows DataLoader workers: the memmaps are opened lazily inside each worker.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from knee.data.series_select import CONTRASTS, PLANES, SLOTS
from knee.paths import LABELS

SPAN = (0.02, 0.98)  # HP: fraction of the ordered stack that window centres cover (public best: 2–98% [KT])
FIELD_STRENGTHS = ["1.5T", "3T", "other"]


def worker_init(_worker_id: int = 0) -> None:
    """DataLoader worker_init_fn: one cv2 thread per worker. cv2's default pool is one thread per core, so N workers
    (× KNEE_IO_THREADS file threads) oversubscribe a 4-vCPU Kaggle box (review 2026-09-30, bug.md #9). The DataLoader
    already sets torch's intra-op threads to 1 in its workers; cv2 has no such default."""
    cv2.setNumThreads(1)


def decode_index(i: int, n: int, epoch: int) -> tuple[int, int]:
    """A dataset index → (epoch, index). Indices ≥ n come from ResumableRandomSampler(encode_epoch=True), which
    folds the epoch into the index so persistent DataLoader workers see it (their dataset copy never gets set_epoch).
    Plain indices keep the dataset's own epoch."""
    return divmod(int(i), n) if i >= n else (epoch, int(i))


# ------------------------------------------------------------------------------------------ window helpers (pure)
def _span_idx(n: int, span: tuple[float, float]) -> tuple[int, int]:
    lo = int(np.floor(span[0] * (n - 1)))
    hi = int(np.ceil(span[1] * (n - 1)))
    return lo, max(lo, hi)


def train_centres(n: int, k: int, rng: np.random.Generator, span=SPAN) -> np.ndarray:
    """k centres over the span: one uniform draw inside each of k equal bins (stratified jitter). Sorted."""
    lo, hi = _span_idx(n, span)
    edges = np.linspace(lo, hi + 1, k + 1)
    c = np.floor(edges[:-1] + rng.random(k) * np.diff(edges)).astype(np.int64)
    return np.clip(c, lo, hi)


def eval_centres(n: int, k: int | None, span=SPAN) -> np.ndarray:
    """k evenly spaced centres over the span; k=None → every slice in the span."""
    lo, hi = _span_idx(n, span)
    if k is None or k >= hi - lo + 1:
        return np.arange(lo, hi + 1, dtype=np.int64)
    return np.rint(np.linspace(lo, hi, k)).astype(np.int64)


def window_indices(n: int, centres: np.ndarray) -> np.ndarray:
    """(k, 3) slice indices [c-1, c, c+1], clamped at the stack ends (edge slice repeated)."""
    c = np.asarray(centres, dtype=np.int64)[:, None]
    return np.clip(c + np.array([-1, 0, 1]), 0, n - 1)


def gather_windows(vol: np.ndarray, centres: np.ndarray, out_px: int | None = None) -> np.ndarray:
    """vol (n, H, W) uint8 (memmap or array) → (k, 3, h, w) uint8, resized with INTER_AREA if out_px is set."""
    idx = window_indices(len(vol), centres)
    uniq, inv = np.unique(idx, return_inverse=True)
    slices = np.asarray(vol[uniq])  # one read per distinct slice (sorted → mostly sequential on disk)
    if out_px is not None and out_px != slices.shape[-1]:
        slices = np.stack([cv2.resize(s, (out_px, out_px), interpolation=cv2.INTER_AREA) for s in slices])
    return slices[inv.reshape(idx.shape)]


def rel_position(n: int, centres: np.ndarray) -> np.ndarray:
    """Normalized position of each centre in its stack: 0 = first slice, 1 = last."""
    return (np.asarray(centres, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)


def type_ids(meta: dict) -> dict[str, int]:
    """Integer ids for the series-type embeddings (plane, fluid, contrast class, field strength, slot)."""
    fs = meta.get("field_strength")
    fs_id = 2 if fs is None or not np.isfinite(fs) else (0 if fs < 2.0 else 1)
    slot = meta.get("slot")
    return {"plane": PLANES.index(meta["plane"]) if meta.get("plane") in PLANES else 0,
            "fluid": int(bool(meta.get("fluid"))),
            "contrast": CONTRASTS.index(meta["contrast_class"]) if meta.get("contrast_class") in CONTRASTS
            else CONTRASTS.index("UNK"),
            "field": fs_id,
            "slot": SLOTS.index(slot) if slot in SLOTS else len(SLOTS)}  # len(SLOTS) = "no slot"


# ------------------------------------------------------------------------------------------ one study → model input
# Pure functions over (pixels, row) pairs: `pixels` is a cache memmap view (training) or ProcessedSeries.pixels from
# preprocess_study (inference), `row` the series metadata (cache index row, or ProcessedSeries.meta + positions_mm).
# Both datasets and knee.infer.predict call these, so train and test windows are assembled by the same code.
TYPE_FIELDS = ("plane", "fluid", "contrast", "field", "slot")


def usable(series: list[tuple[np.ndarray, dict]]) -> list[tuple[np.ndarray, dict]]:
    """Drop series with no slices or with no slice that decoded (`valid` all False / n_decode_fail = n): their pixels
    are all zeros, so they would only add black windows (review 2026-09-30). A partly failed series is kept."""
    out = []
    for vol, row in series:
        n = len(vol)
        valid = row.get("valid")
        n_ok = int(np.sum(valid)) if valid is not None and len(valid) == n else n - int(row.get("n_decode_fail") or 0)
        if n > 0 and n_ok > 0:
            out.append((vol, row))
    return out


def slots_sample(series: list[tuple[np.ndarray, dict]], W: int, px: int, train: bool, rng: np.random.Generator,
                 span=SPAN, slot_dropout: float = 0.0) -> dict[str, np.ndarray]:
    """Track A grid: x (5, W, 3, px, px) uint8, slot_mask (5,), z (5, W). Missing slots are zeros, mask False."""
    by_slot = {row["slot"]: (vol, row) for vol, row in usable(series) if row.get("slot") in SLOTS}
    x = np.zeros((len(SLOTS), W, 3, px, px), dtype=np.uint8)
    z = np.zeros((len(SLOTS), W), dtype=np.float32)
    mask = np.zeros(len(SLOTS), dtype=bool)
    present = [k for k, s in enumerate(SLOTS) if s in by_slot]
    if train and slot_dropout > 0 and len(present) > 1:
        drop = rng.random(len(present)) < slot_dropout
        if drop.all():
            drop[rng.integers(len(present))] = False
        present = [k for k, d in zip(present, drop) if not d]
    for k in present:
        vol, _ = by_slot[SLOTS[k]]
        n = len(vol)
        c = train_centres(n, W, rng, span) if train else eval_centres(n, W, span)
        if len(c) < W:  # stack shorter than W: repeat evenly (eval) so the grid stays fixed
            c = c[np.rint(np.linspace(0, len(c) - 1, W)).astype(int)]
        x[k] = gather_windows(vol, c, px)
        z[k] = rel_position(n, c)
        mask[k] = True
    return {"x": x, "slot_mask": mask, "z": z}


def bag_sample(series: list[tuple[np.ndarray, dict]], K: int, K_eval: int | None, px: int, train: bool,
               rng: np.random.Generator, span=SPAN, series_dropout: float = 0.0,
               max_series: int | None = None) -> dict[str, np.ndarray]:
    """Track B bag: x (L, 3, px, px) uint8 + per-window series_idx, z, z_mm and type ids (L,). L = 0 if no series.

    max_series (eval): the slotted series first, in slot order, then the rest in their given order. Folder order
    alone is UID order, i.e. arbitrary."""
    series = usable(series)
    if train and series_dropout > 0 and len(series) > 1:
        keep = rng.random(len(series)) >= series_dropout
        if not keep.any():
            keep[rng.integers(len(series))] = True
        series = [s for s, k in zip(series, keep) if k]
    if max_series is not None and len(series) > max_series:
        if train:
            pick = rng.choice(len(series), max_series, replace=False)
        else:
            rank = [SLOTS.index(r.get("slot")) if r.get("slot") in SLOTS else len(SLOTS) for _, r in series]
            pick = sorted(range(len(series)), key=lambda j: (rank[j], j))[:max_series]
        series = [series[j] for j in sorted(pick)]
    xs, sid, z, zmm = [np.zeros((0, 3, px, px), np.uint8)], [np.zeros(0, np.int64)], [np.zeros(0, np.float32)], \
        [np.zeros(0, np.float32)]
    tid = {k: [np.zeros(0, np.int64)] for k in TYPE_FIELDS}
    for j, (vol, row) in enumerate(series):
        n = len(vol)
        c = train_centres(n, K, rng, span) if train else eval_centres(n, K_eval, span)
        xs.append(gather_windows(vol, c, px))
        sid.append(np.full(len(c), j, dtype=np.int64))
        z.append(rel_position(n, c))
        pos = np.asarray(row["positions_mm"], dtype=np.float32)
        zmm.append(pos[c] - np.float32(np.median(pos)))  # mm from the stack's middle (optional feature)
        for k, v in type_ids(row).items():
            tid[k].append(np.full(len(c), v, dtype=np.int64))
    out = {"x": np.concatenate(xs), "series_idx": np.concatenate(sid), "z": np.concatenate(z),
           "z_mm": np.concatenate(zmm)}
    out.update({k: np.concatenate(v) for k, v in tid.items()})
    return out


# ------------------------------------------------------------------------------------------ cache access
class CacheReader:
    """Read-only view of a built cache. Picklable: memmaps are (re)opened lazily in each process."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
        self.out_px = int(self.manifest["params"]["out_px"])
        ser = pd.read_parquet(self.cache_dir / "series_index.parquet")
        self.studies = pd.read_parquet(self.cache_dir / "study_index.parquet").set_index("study_uid")
        self._series_by_study: dict[str, list[dict]] = {
            uid: g.sort_values("offset").to_dict("records") for uid, g in ser.groupby("study_uid", sort=False)}
        self._mm: dict[int, np.memmap] = {}

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_mm"] = {}
        return d

    @property
    def study_uids(self) -> list[str]:
        return sorted(self._series_by_study)

    def series(self, study_uid: str) -> list[dict]:
        return self._series_by_study.get(study_uid, [])

    def _shard(self, shard: int) -> np.memmap:
        mm = self._mm.get(shard)
        if mm is None:
            f = self.cache_dir / "shards" / f"shard_{shard:05d}.u8"
            n = f.stat().st_size // (self.out_px * self.out_px)
            mm = np.memmap(f, dtype=np.uint8, mode="r", shape=(n, self.out_px, self.out_px))
            self._mm[shard] = mm
        return mm

    def pixels(self, row: dict) -> np.ndarray:
        """(n_slices, out_px, out_px) uint8 memmap view of one series."""
        o = int(row["offset"])
        return self._shard(int(row["shard"]))[o: o + int(row["n_slices"])]


def _labels_for(labels: pd.DataFrame | None, uid: str) -> tuple[torch.Tensor, torch.Tensor]:
    if labels is None or uid not in labels.index:
        return torch.zeros(len(LABELS)), torch.zeros(len(LABELS), dtype=torch.bool)
    v = labels.loc[uid, LABELS].to_numpy(dtype=np.float32)
    m = np.isfinite(v)
    return torch.from_numpy(np.nan_to_num(v)), torch.from_numpy(m)


# ------------------------------------------------------------------------------------------ Track A
class StudyWindowsDataset(Dataset):
    """Track A: a fixed grid of 5 slots × W windows per study.

    Returns x uint8 (S=5, W, 3, h, w), slot_mask (S,), z (S, W) relative positions, labels (12,), label_mask (12,).
    Missing slots are zeros with slot_mask False. train=True draws random centres (and optional slot dropout).
    """

    def __init__(self, reader: CacheReader, study_uids: list[str], labels: pd.DataFrame | None = None,
                 windows_per_slot: int = 12, train: bool = False, out_px: int | None = None, span=SPAN,
                 slot_dropout: float = 0.0, seed: int = 0):
        self.r, self.uids, self.labels = reader, list(study_uids), labels
        self.W, self.train, self.span, self.slot_dropout, self.seed = windows_per_slot, train, span, slot_dropout, seed
        self.px = out_px or reader.out_px
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int) -> dict:
        epoch, i = decode_index(i, len(self.uids), self.epoch)
        uid = self.uids[i]
        rng = np.random.default_rng((self.seed, epoch, i))
        series = [(self.r.pixels(s), s) for s in self.r.series(uid)]
        out = slots_sample(series, self.W, self.px, self.train, rng, self.span, self.slot_dropout)
        y, ym = _labels_for(self.labels, uid)
        return {**{k: torch.from_numpy(v) for k, v in out.items()}, "labels": y, "label_mask": ym, "study_uid": uid}


# ------------------------------------------------------------------------------------------ Track B
class StudyBagDataset(Dataset):
    """Track B: a variable-length bag of windows from every series of the study (not only the 5 slots).

    Each window carries its series' type ids and its slice position. Use collate_bag to pad a batch.
    train: windows_per_series random windows per series (+ optional series dropout).
    eval:  eval_windows_per_series evenly spaced (None = every slice in the span).
    """

    def __init__(self, reader: CacheReader, study_uids: list[str], labels: pd.DataFrame | None = None,
                 windows_per_series: int = 8, eval_windows_per_series: int | None = 16, train: bool = False,
                 out_px: int | None = None, span=SPAN, series_dropout: float = 0.0, max_series: int | None = None,
                 seed: int = 0):
        self.r, self.uids, self.labels = reader, list(study_uids), labels
        self.K, self.K_eval, self.train, self.span = windows_per_series, eval_windows_per_series, train, span
        self.series_dropout, self.max_series, self.seed = series_dropout, max_series, seed
        self.px = out_px or reader.out_px
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int) -> dict:
        epoch, i = decode_index(i, len(self.uids), self.epoch)
        uid = self.uids[i]
        rng = np.random.default_rng((self.seed, epoch, i))
        series = [(self.r.pixels(s), s) for s in self.r.series(uid)]
        out = bag_sample(series, self.K, self.K_eval, self.px, self.train, rng, self.span, self.series_dropout,
                         self.max_series)
        y, ym = _labels_for(self.labels, uid)
        return {**{k: torch.from_numpy(v) for k, v in out.items()}, "labels": y, "label_mask": ym, "study_uid": uid}


def collate_bag(batch: list[dict]) -> dict:
    """Pad bags to the longest one; window_mask marks real windows (True) vs padding (False)."""
    L = max(len(b["x"]) for b in batch)
    B = len(batch)
    out: dict = {"window_mask": torch.zeros(B, L, dtype=torch.bool), "study_uid": [b["study_uid"] for b in batch],
                 "labels": torch.stack([b["labels"] for b in batch]),
                 "label_mask": torch.stack([b["label_mask"] for b in batch])}
    for k in ("x", "series_idx", "z", "z_mm", "plane", "fluid", "contrast", "field", "slot"):
        ref = batch[0][k]
        t = torch.zeros((B, L, *ref.shape[1:]), dtype=ref.dtype)
        for i, b in enumerate(batch):
            t[i, : len(b[k])] = b[k]
        out[k] = t
    for i, b in enumerate(batch):
        out["window_mask"][i, : len(b["x"])] = True
    return out
