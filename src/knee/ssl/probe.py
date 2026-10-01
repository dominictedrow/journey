"""Frozen-feature linear probe for the MAE encoder (02 §D "monitoring"): are the features getting better, not just
the reconstruction loss?

Per study: evenly spaced eval windows from every series → frozen SliceViT → mean window vector per plane
(sagittal, coronal, axial; zeros if a plane is missing) → a 3E study vector. One logistic regression per finding,
trained on the other folds' studies and scored on the probe fold (binarized report labels). A cheap, relative signal:
compare it across checkpoints of the same run, not with fine-tuned numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from knee.data.dataset import CacheReader, StudyBagDataset
from knee.model.slice_vit import SliceViT, to_input
from knee.paths import LABELS
from knee.utils.metrics import binarize, macro_auc


@torch.no_grad()
def study_features(encoder: SliceViT, reader: CacheReader, uids: list[str], device, out_px: int,
                   windows_per_series: int = 8, chunk: int = 128, autocast=None) -> np.ndarray:
    ds = StudyBagDataset(reader, uids, eval_windows_per_series=windows_per_series, train=False, out_px=out_px)
    encoder.eval()
    feats = np.zeros((len(uids), 3 * encoder.dim), dtype=np.float32)
    for i in range(len(ds)):
        b = ds[i]
        x = b["x"].to(device)
        with autocast if autocast is not None else torch.autocast(device.type, enabled=False):
            v = torch.cat([encoder(to_input(c)) for c in x.split(chunk)]).float().cpu().numpy()
        plane = b["plane"].numpy()
        for p in range(3):
            if (plane == p).any():
                feats[i, p * encoder.dim:(p + 1) * encoder.dim] = v[plane == p].mean(0)
    return feats


def linear_probe(encoder: SliceViT, reader: CacheReader, labels: pd.DataFrame, folds: pd.DataFrame, device,
                 out_px: int, probe_fold: int = 0, max_train: int = 1500, max_val: int = 600, seed: int = 0,
                 autocast=None) -> dict:
    """labels: index study uid, the 12 LABELS columns (soft). folds: index study uid, column 'fold'."""
    have = set(reader.study_uids) & set(labels.index) & set(folds.index)
    rng = np.random.default_rng(seed)
    tr = sorted(u for u in have if folds.loc[u, "fold"] != probe_fold)
    va = sorted(u for u in have if folds.loc[u, "fold"] == probe_fold)
    tr = sorted(rng.choice(tr, min(max_train, len(tr)), replace=False).tolist())
    va = sorted(rng.choice(va, min(max_val, len(va)), replace=False).tolist())
    f_tr = study_features(encoder, reader, tr, device, out_px, autocast=autocast)
    f_va = study_features(encoder, reader, va, device, out_px, autocast=autocast)
    sc = StandardScaler().fit(f_tr)
    f_tr, f_va = sc.transform(f_tr), sc.transform(f_va)
    y_tr, y_va = binarize(labels.loc[tr, LABELS].to_numpy()), binarize(labels.loc[va, LABELS].to_numpy())
    p = np.full(y_va.shape, np.nan)
    for k in range(len(LABELS)):
        ok = np.isfinite(y_tr[:, k])
        if ok.sum() > 10 and 0 < y_tr[ok, k].sum() < ok.sum():
            clf = LogisticRegression(C=0.1, max_iter=2000).fit(f_tr[ok], y_tr[ok, k])
            p[:, k] = clf.predict_proba(f_va)[:, 1]
    auc, n = macro_auc(y_va, p, "ignore")  # a finding with too few train labels has no probe: its column is NaN
    return {"probe_macro_auc": auc, "probe_n_findings": n, "probe_n_train": len(tr), "probe_n_val": len(va)}
