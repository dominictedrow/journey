"""Round trip of the Kaggle cache upload layout (AB.md Part 1): stage_kaggle_cache splits the cache into flat shard
datasets + an index dataset; knee.kaggle_farm rebuilds one CacheReader root from them, and every pixel matches."""
import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np

from fake_cache import make_fake_cache
from knee.data.dataset import CacheReader
from knee.kaggle_farm import build_farm

REPO = Path(__file__).resolve().parents[1]


def test_stage_then_farm_roundtrip(tmp_path, monkeypatch):
    f = make_fake_cache(tmp_path / "c", n_studies=6, px=16)
    lab = tmp_path / "labels"
    lab.mkdir()
    shutil.copy(f["labels"], lab / "labels_v0.parquet")
    shutil.copy(f["folds"], lab / "folds_v0.csv")
    spec = importlib.util.spec_from_file_location("stage", REPO / "scripts" / "stage_kaggle_cache.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    stage.stage(f["cache"], lab, tmp_path / "stage", parts=1, user="u")
    assert all(p.is_file() for d in (tmp_path / "stage").iterdir() for p in d.iterdir())  # flat: -r skip drops subdirs

    monkeypatch.setattr(Path, "symlink_to", lambda self, t: os.link(t, self))  # Windows needs admin for symlinks
    root, idx = build_farm(tmp_path / "farm", parts=1, roots=[str(tmp_path / "stage" / "{s}")])
    assert (idx / "folds_v0.csv").exists() and (idx / "trackA_coatnet.yaml").exists()
    a, b = CacheReader(f["cache"]), CacheReader(root)
    assert a.study_uids == b.study_uids
    for u in a.study_uids:
        for sa, sb in zip(a.series(u), b.series(u)):
            np.testing.assert_array_equal(np.asarray(a.pixels(sa)), np.asarray(b.pixels(sb)))
