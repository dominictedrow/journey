"""A tiny cache in the S02 on-disk format (manifest + raw uint8 shards + parquet indexes), plus matching labels and
folds files, for CPU smoke tests of the MAE and supervised trainers. Pixels are random with a per-study brightness
pattern so a model has something weakly learnable."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from knee.data.series_select import SLOTS
from knee.paths import LABELS

SERIES_TYPES = [("Sagittal", True, "PD", "sag_fluid"), ("Sagittal", False, "T1", "sag_other"),
                ("Coronal", True, "PD", "cor_fluid"), ("Coronal", False, "PD", "cor_other"),
                ("Axial", True, "T2", "axial")]


def make_fake_cache(root: Path, n_studies: int = 10, px: int = 64, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "shards").mkdir(parents=True, exist_ok=True)
    ser_rows, stu_rows, lab_rows = [], [], []
    offset = 0
    with open(root / "shards" / "shard_00000.u8", "wb") as fh:
        for i in range(n_studies):
            uid = f"1.2.3.{i}"
            y = rng.random(len(LABELS)) < 0.4
            n_series = 3 + i % 3
            for j in range(n_series):
                plane, fluid, contrast, slot = SERIES_TYPES[j]
                n = int(rng.integers(5, 12))
                vol = rng.integers(0, 80, (n, px, px), dtype=np.uint8)
                vol[:, : px // 4, : px // 4] += np.uint8(100 * y[j % len(LABELS)])  # a weak label signal
                fh.write(vol.tobytes())
                ser_rows.append({"study_uid": uid, "shard": 0, "offset": offset, "n_slices": n,
                                 "series_uid": f"{uid}.{j}", "plane": plane, "fluid": fluid,
                                 "contrast_class": contrast, "field_strength": 3.0 if i % 2 else 1.5,
                                 "slot": slot if slot in SLOTS else None,
                                 "positions_mm": (np.arange(n, dtype=np.float32) * 3.0).tolist(),
                                 "valid": [True] * n})
                offset += n
            stu_rows.append({"study_uid": uid, "shard": 0, "n_series": n_series, "error": None})
            soft = np.where(y, rng.uniform(0.7, 1.0, len(LABELS)), rng.uniform(0.0, 0.3, len(LABELS)))
            lab_rows.append({"StudyInstanceUID": uid, **dict(zip(LABELS, soft)),
                             **{f"w_{k}": 1.0 for k in LABELS}})
    pd.DataFrame(ser_rows).to_parquet(root / "series_index.parquet", index=False)
    pd.DataFrame(stu_rows).to_parquet(root / "study_index.parquet", index=False)
    (root / "cache_manifest.json").write_text(json.dumps({"version": "fake", "params": {"out_px": px}}))
    labels = pd.DataFrame(lab_rows)
    labels.to_parquet(root / "labels.parquet", index=False)
    pd.DataFrame({"StudyInstanceUID": labels.StudyInstanceUID, "fold": np.arange(n_studies) % 2}).to_csv(
        root / "folds.csv", index=False)
    gold = labels.iloc[:4][["StudyInstanceUID", *LABELS]].copy()
    gold[LABELS] = (gold[LABELS] > 0.5).astype(int)
    gold.to_csv(root / "gold.csv", index=False)
    return {"cache": root, "labels": root / "labels.parquet", "folds": root / "folds.csv", "gold": root / "gold.csv"}
