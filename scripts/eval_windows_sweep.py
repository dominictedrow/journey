"""Re-score a trained Track A fold at several eval window counts per slot (no retraining; E017).

    python scripts/eval_windows_sweep.py --fold 0 --windows 12 16 20 24

Scores against labels_v0 (as in training) and against labels_v0 with silent cells set to 0.15 (E016: silent ≈ negative
on gold), which also scores the cells v0 masks. OOF per W goes to <run>/sweep/oof_w<W>.parquet.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from knee.data.dataset import CacheReader, StudyWindowsDataset
from knee.infer.predict import load_arm
from knee.labels import public_sets as PS
from knee.labels.blend import BlendConfig, blend
from knee.paths import LABELS, load_paths
from knee.train.train_sup import evaluate, load_gold, predict
from knee.utils.metrics import binarize, macro_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--windows", type=int, nargs="+", default=[12, 16, 20, 24])
    ap.add_argument("--run", default="C:/knee_cache/trackA_runs")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    P = load_paths()
    run = Path(a.run) / f"fold{a.fold}"
    labels = pd.read_parquet(P.cache_root / "labels/labels_v0.parquet").set_index("StudyInstanceUID")
    folds = pd.read_csv(P.cache_root / "labels/folds_v0.csv").set_index("StudyInstanceUID")
    gold = load_gold(P.train_csv)
    reader = CacheReader(P.cache_root / "cache_v1_train")
    uids = sorted(u for u in set(reader.study_uids) & set(labels.index) if folds.loc[u, "fold"] == a.fold)

    # silent-fixed labels (E016), recomputed from the same readers/config that built v0
    frames = {n: PS.load(n, PS.public_dir(P.cache_root)) for n in PS.DEFAULT_READERS}
    _, _, status = blend(frames, labels.index.tolist(), BlendConfig())
    fixed = labels[LABELS].mask(status == "silent", 0.15)

    device = torch.device("cuda")
    model, dcfg = load_arm(run / "model_avg.pt", device, {"channels_last": True})
    out = run / "sweep"
    out.mkdir(exist_ok=True)
    for W in a.windows:
        ds = StudyWindowsDataset(reader, uids, labels, windows_per_slot=W, train=False, out_px=dcfg["out_px"])
        u, p, st = predict(model, ds, None, device, "bf16", a.workers, batch_size=2)
        r = evaluate(u, p, labels, gold, 0.4, 0.6, st["n_retried_fp32"])
        fx, _ = macro_auc(binarize(fixed.loc[u].to_numpy(), 0.4, 0.6), p)
        pd.DataFrame(p, columns=LABELS).assign(StudyInstanceUID=u).to_parquet(out / f"oof_w{W}.parquet", index=False)
        print(f"fold {a.fold} W {W:3d} ({5 * W} windows): val {r['val_macro_auc']:.4f}  val_silentfix {fx:.4f}  "
              f"gold {r.get('gold_macro_auc', np.nan):.3f} (n {r.get('gold_n', 0)})  nonfinite {st['n_nonfinite']}", flush=True)


if __name__ == "__main__":
    main()
