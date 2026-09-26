"""5-fold split for every track (S04 step 8, master plan §6).

Groups: studies that share a **normalized report hash** (csv_audit.report_hash; 52 groups / 223 studies in S01) always
land in the same fold, so a duplicated report can't leak its labels across folds. With `site_mode="group"`, the
site proxy (scanner fingerprint from S01 part 2, `<audit_out>/study_train.parquet`) is merged into the groups too
(union-find), i.e. whole sites are held out: a harsher, domain-shift-style split.

Stratification (StratifiedGroupKFold on one composite key): gold studies form their own stratum so the 58 spread
evenly (~11–12 per fold); the other studies are stratified on a few binarized blended labels (the rarer, harder ones)
and, with `site_mode="stratify"` (the default when the site proxy exists), on the site too, so every fold sees
every scanner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from knee.audit.csv_audit import report_hash

STRAT_LABELS = ["ACL", "Lateral Meniscus", "Fracture", "Lateral OA"]


def _union_groups(keys: list[pd.Series]) -> np.ndarray:
    """Connected components over studies linked by any shared key value."""
    n = len(keys[0])
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key in keys:
        first: dict = {}
        for i, v in enumerate(key.to_numpy()):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if v in first:
                a, b = find(i), find(first[v])
                if a != b:
                    parent[a] = b
            else:
                first[v] = i
    return np.array([find(i) for i in range(n)])


def make_folds(train: pd.DataFrame, p: pd.DataFrame, n_folds: int = 5, seed: int = 42,
               site: pd.Series | None = None, site_mode: str = "stratify") -> pd.DataFrame:
    """train: train.csv (StudyInstanceUID, Report, labels); p: blended soft labels indexed by StudyInstanceUID.
    site: study → site proxy (optional). Returns StudyInstanceUID, fold, report_hash, group, stratum."""
    from knee.paths import LABELS
    df = train[["StudyInstanceUID", "Report"]].copy().reset_index(drop=True)
    df["report_hash"] = df.Report.map(report_hash)
    gold = train[LABELS].notna().all(axis=1).to_numpy()
    keys = [df.report_hash]
    s = site.reindex(df.StudyInstanceUID).reset_index(drop=True) if site is not None else None
    if s is not None and site_mode == "group":
        keys.append(s)
    df["group"] = _union_groups(keys)
    if df.group.nunique() < 2 * n_folds:
        raise ValueError(f"only {df.group.nunique()} groups after merging keys: report duplicates chain sites "
                         "together; use site_mode='stratify'")

    lab = (p.reindex(df.StudyInstanceUID)[STRAT_LABELS].to_numpy() >= 0.5).astype(int)
    code = (lab * (2 ** np.arange(lab.shape[1]))).sum(axis=1)
    stratum = pd.Series(code.astype(str))
    if s is not None and site_mode == "stratify":
        top = s.value_counts()
        s_small = s.where(s.map(top).fillna(0) >= 2 * n_folds, "other").fillna("unknown")
        stratum = stratum + "|" + s_small.astype(str)
    stratum[gold] = "gold"
    df["stratum"] = stratum.to_numpy()
    y = pd.factorize(df.stratum)[0]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for f, (_, va) in enumerate(sgkf.split(df, y, groups=df.group)):
        df.loc[va, "fold"] = f
    assert (df.fold >= 0).all()
    return df[["StudyInstanceUID", "fold", "report_hash", "group", "stratum"]]


def check_folds(folds: pd.DataFrame) -> dict:
    """Invariants: every group in exactly one fold; returns sizes for the QA report."""
    per_group = folds.groupby("group").fold.nunique()
    assert (per_group == 1).all(), "a report-hash group spans folds"
    per_hash = folds.groupby("report_hash").fold.nunique()
    assert (per_hash == 1).all(), "a duplicated report spans folds"
    return {"sizes": folds.fold.value_counts().sort_index().tolist(),
            "gold_per_fold": folds[folds.stratum == "gold"].fold.value_counts().sort_index().tolist(),
            "n_groups": int(folds.group.nunique()),
            "multi_study_groups": int((folds.group.value_counts() > 1).sum())}
