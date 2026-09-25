"""Eyeball figures and DataLoader throughput for a built cache (S02 "Measure" + "Tests").

    python -m knee.data.inspect_cache plot  --cache C:/knee_cache/cache_v1_train --n 3 --out docs/figures
    python -m knee.data.inspect_cache bench --cache C:/knee_cache/cache_v1_train --workers 4 --batches 30

plot: for n random studies, one row per slot × 5 evenly spaced windows (centre slice). Check by eye: every knee is a
RIGHT knee (coronal/axial: medial = fibula-free side on the image RIGHT; sagittal: slice 0 medial), head at the top,
anterior on the left in sagittal, the joint centred, and no slot shows the wrong plane or contrast.
bench: studies/s through a real DataLoader for both datasets at the planned batch shapes.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def plot(cache: Path, n: int, out: Path, seed: int = 0, windows: int = 5) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from knee.data.dataset import CacheReader, eval_centres
    from knee.data.series_select import SLOTS

    r = CacheReader(cache)
    rng = np.random.default_rng(seed)
    uids = list(rng.choice(r.study_uids, size=min(n, len(r.study_uids)), replace=False))
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, uid in enumerate(uids):
        st = r.studies.loc[uid]
        by_slot = {s["slot"]: s for s in r.series(uid) if s.get("slot")}
        fig, axes = plt.subplots(len(SLOTS), windows, figsize=(windows * 1.6, len(SLOTS) * 1.7))
        for i, slot in enumerate(SLOTS):
            row = by_slot.get(slot)
            vol = r.pixels(row) if row is not None else None
            cs = eval_centres(len(vol), windows) if vol is not None else []
            for j in range(windows):
                ax = axes[i, j]
                ax.set_xticks([]), ax.set_yticks([])
                if vol is not None and j < len(cs):
                    ax.imshow(vol[cs[j]], cmap="gray", vmin=0, vmax=255)
                    ax.set_title(f"slice {cs[j]}/{len(vol) - 1}", fontsize=6)
                if j == 0:
                    lab = slot if row is None else f"{slot}\n{row['contrast_class']} fs={int(row['fluid'])}"
                    ax.set_ylabel(lab, fontsize=7)
        fig.suptitle(f"study #{k} | laterality {st['laterality']} ({st['laterality_source']}), "
                     f"mirrored={bool(st['mirrored'])} | shown as a RIGHT knee", fontsize=8)
        fig.tight_layout()
        p = out / f"cache_check_{k}.png"
        fig.savefig(p, dpi=70)
        plt.close(fig)
        paths.append(p)
    return paths


def bench(cache: Path, workers: int, batches: int, out_px: int | None) -> dict:
    from torch.utils.data import DataLoader

    from knee.data.dataset import CacheReader, StudyBagDataset, StudyWindowsDataset, collate_bag

    r = CacheReader(cache)
    uids = r.study_uids
    res = {}
    cfgs = {
        "trackA_train (bs2, 5 slots x 12 win)": (StudyWindowsDataset(r, uids, windows_per_slot=12, train=True,
                                                                     out_px=out_px), 2, None),
        "trackA_eval (bs1, 5 slots x 16 win)": (StudyWindowsDataset(r, uids, windows_per_slot=16, out_px=out_px),
                                                1, None),
        "trackB_train (bs4, 8 win/series)": (StudyBagDataset(r, uids, windows_per_series=8, train=True,
                                                             out_px=out_px), 4, collate_bag),
    }
    for name, (ds, bs, coll) in cfgs.items():
        dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers, collate_fn=coll,
                        persistent_workers=workers > 0, drop_last=False)
        n_studies, t0, first = 0, None, None
        it = 0
        while it < batches:
            for b in dl:
                if t0 is None:
                    first = time.time()
                    t0 = first
                    continue  # exclude worker start-up from the rate
                n_studies += len(b["study_uid"])
                it += 1
                if it >= batches:
                    break
        el = time.time() - t0 if t0 else float("nan")
        res[name] = {"studies_per_s": n_studies / el if el else float("nan"), "batches": it,
                     "x_shape": tuple(b["x"].shape)}
        print(f"{name}: {res[name]['studies_per_s']:.1f} studies/s, x {res[name]['x_shape']}", flush=True)
        del dl
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plot", "bench"])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--out-px", type=int, default=None, help="resize at load (default: cache size)")
    a = ap.parse_args()
    if a.cmd == "plot":
        for p in plot(a.cache, a.n, a.out, a.seed):
            print(p)
    else:
        bench(a.cache, a.workers, a.batches, a.out_px)


if __name__ == "__main__":
    main()
