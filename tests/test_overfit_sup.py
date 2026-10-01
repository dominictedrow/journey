"""Overfit test through the real supervised path: labels parquet → StudyBagDataset (_labels_for: NaN → label_mask)
→ collate_bag → KneeViT → soft_bce (+ rank loss) with the w_<label> weights → train_sup's loop → model_avg.pt →
train_sup.predict. test_model.py::test_overfit_8_studies only checks the model can learn a hand-built batch with a
plain BCE; this one catches wiring bugs in the labels, the label mask, the weights, the loss and the head.

Every slice of a study is the same image, so any window identifies the study and 8 studies can be memorized.
Two "twin" studies copy another study's series rows (same pixels, same metadata → the model gives identical outputs)
and conflict with it only on a masked (NaN) or zero-weight entry. If the mask or the weights were ignored, those
outputs would be pulled to ~0.5 instead of matching the twin's real label.
"""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from fake_cache import make_fake_cache  # noqa: E402
from knee.data.dataset import CacheReader  # noqa: E402
from knee.paths import LABELS  # noqa: E402
from knee.train import train_sup  # noqa: E402

TINY_VIT = {"img_size": 32, "patch": 8, "dim": 32, "depth": 2, "heads": 2, "drop_path": 0.0}
N = 8
TWIN_MASK = (0, 6, 2)    # (source study, twin study, label index): the twin's label is NaN
TWIN_WEIGHT = (1, 7, 5)  # the twin's label is the opposite, with weight 0


def _make_memorizable_cache(root, px=32, seed=0):
    f = make_fake_cache(root, n_studies=N, px=px, seed=seed)
    cache = f["cache"]
    ser = pd.read_parquet(cache / "series_index.parquet")
    # every slice of a study = one random image per study (any window identifies the study)
    shard = cache / "shards" / "shard_00000.u8"
    mm = np.memmap(shard, dtype=np.uint8, mode="r+").reshape(-1, px, px)
    rng = np.random.default_rng(seed + 1)
    for uid, g in ser.groupby("study_uid"):
        img = rng.integers(0, 256, (px, px), dtype=np.uint8)
        for _, r in g.iterrows():
            mm[r.offset: r.offset + r.n_slices] = img
    mm.flush()
    del mm
    # twins: replace the twin's series rows with the source's rows (same shard offsets → same pixels and metadata)
    for src, twin, _ in (TWIN_MASK, TWIN_WEIGHT):
        s_uid, t_uid = f"1.2.3.{src}", f"1.2.3.{twin}"
        rows = ser[ser.study_uid == s_uid].copy()
        rows["study_uid"] = t_uid
        rows["series_uid"] = rows.series_uid.str.replace(s_uid, t_uid, regex=False)
        ser = pd.concat([ser[ser.study_uid != t_uid], rows], ignore_index=True)
        stu = pd.read_parquet(cache / "study_index.parquet")
        stu.loc[stu.study_uid == t_uid, "n_series"] = len(rows)
        stu.to_parquet(cache / "study_index.parquet", index=False)
    ser.to_parquet(cache / "series_index.parquet", index=False)

    # hard 0/1 labels, both classes in every column; ~15% masked; weights 1 except the zero-weight twin entry
    y = (rng.random((N, len(LABELS))) < 0.5).astype(np.float64)
    y[0, :], y[1, :] = 1 - y[1, :], y[1, :]
    y[2:, 0] = [1, 0, 1, 0, 1, 0]
    w = np.ones_like(y)
    nan = rng.random(y.shape) < 0.15
    for src, twin, _ in (TWIN_MASK, TWIN_WEIGHT):
        y[twin], nan[twin], nan[src] = y[src], nan[src], nan[src]  # twins agree everywhere else
    s, t, k = TWIN_MASK
    nan[s, k], y[s, k], nan[t, k] = False, 1.0, True
    s, t, k = TWIN_WEIGHT
    nan[s, k], nan[t, k] = False, False
    y[s, k], y[t, k], w[t, k] = 1.0, 0.0, 0.0
    yl = np.where(nan, np.nan, y)
    uids = [f"1.2.3.{i}" for i in range(N)]
    df = pd.DataFrame({"StudyInstanceUID": uids, **{k: yl[:, j] for j, k in enumerate(LABELS)},
                       **{f"w_{k}": w[:, j] for j, k in enumerate(LABELS)}})
    cols = list(df.columns[1:])
    np.random.default_rng(3).shuffle(cols)  # labels are read by name, not position: shuffle the column order
    df[["StudyInstanceUID", *cols]].to_parquet(f["labels"], index=False)
    return f, uids, yl, w


@pytest.mark.parametrize("rank_w", [0.0, 0.05])
def test_train_sup_overfits_8_studies(tmp_path, rank_w):
    torch.manual_seed(0)
    f, uids, y, w = _make_memorizable_cache(tmp_path / "c")
    cfg = {"run_name": "overfit", "cache_dir": str(f["cache"]), "run_dir": str(tmp_path / "run"), "device": "cpu",
           "precision": "fp32", "seed": 0,
           "model": {"kind": "knee_vit", "cfg": {"slice_vit": TINY_VIT, "window_dropout": 0.0}},
           "labels": {"file": str(f["labels"])}, "folds": {"file": str(f["folds"]), "fold": -1},
           "data": {"out_px": 32, "train_windows": 2, "eval_windows": 3, "batch_size": 2, "workers": 0,
                    "series_dropout": 0.0},
           "optim": {"epochs": 150, "lr": 3e-3, "weight_decay": 0.0, "warmup_frac": 0.05, "min_lr_ratio": 0.1,
                     "accum": 1},
           "loss": {"rank_w": rank_w}, "select": {"top_k": 1}, "log_every": 4, "ckpt_minutes": 1e9}
    cp = tmp_path / "cfg.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    res = train_sup.main(["--config", str(cp)])
    assert res["top_epochs"] == [149]

    # 1. the training loss (masked, weighted soft BCE [+ rank]) goes to ~0
    losses = [r["loss"] for r in map(json.loads, (tmp_path / "run" / "log.jsonl").read_text().splitlines())
              if "loss" in r]
    assert losses[0] > 0.5, losses[:3]           # starts near ln 2 (zero-init head)
    assert losses[-1] < 0.02, losses[-5:]

    # 2. the saved averaged model, run through the eval path, reproduces every labelled entry
    ck = torch.load(tmp_path / "run" / "model_avg.pt", map_location="cpu", weights_only=False)
    model = train_sup.build_model(ck["cfg"]["model"])
    model.load_state_dict(ck["model"])
    labels = pd.read_parquet(f["labels"]).set_index("StudyInstanceUID")
    ds, collate = train_sup.make_dataset(cfg | {"data": {**train_sup.DEFAULTS["data"], **cfg["data"]}},
                                         CacheReader(f["cache"]), uids, labels, train=False)
    got_uids, p = train_sup.predict(model, ds, collate, torch.device("cpu"), "fp32", 0, batch_size=3)
    assert got_uids == uids
    used = np.isfinite(y) & (w > 0)
    err = np.abs(p - np.nan_to_num(y))[used]
    assert err.max() < 0.1, (err.max(), np.argwhere(used & (np.abs(p - np.nan_to_num(y)) >= 0.1)))

    # 3. the twins: the masked / zero-weight conflicting entry did not pull the shared output toward 0.5
    for src, twin, k in (TWIN_MASK, TWIN_WEIGHT):
        np.testing.assert_allclose(p[src], p[twin], atol=1e-5)  # identical inputs → identical outputs
        assert p[src, k] > 0.9, (LABELS[k], p[src, k])
