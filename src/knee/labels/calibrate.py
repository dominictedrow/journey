"""Tiny per-finding calibration of our reader's categories to the gold annotators' policy (S04 step 7).

Our reader outputs a bin per (study, finding): absent < absent_degenerative < uncertain < present_1 < present_2 <
present_3, or None (not mentioned). This module learns, per finding, what each bin is worth:

    value(bin) = (positives_in_bin + m · prior(bin)) / (n_in_bin + m)        (shrink toward prompt.PRIOR)
    then isotonic (non-decreasing) over the bin order, weighted by n + m.

That is at most 6 numbers per finding, shrunk hard (m = 5 pseudo-studies), and monotone, so it can only merge
neighbouring bins, never reorder them. It exists to set **target values** (e.g. a small effusion is worth ~0.3,
not ~0.65, if the gold readers mostly call it negative), not to chase gold AUC. Per-finding choices tuned on the 58
have hurt on the LB before (docs/kaggle/labels_from_notebooks.md §3).

**Out of fold:** each gold study's calibrated value comes from a map fitted on the other 57 (leave-one-out), so the
gold AUC reported for "calibrated" is honest. Non-gold studies use the map fitted on all 58.

Known bias: the gold 58 are enriched (every study has ≥ 1 positive; Fracture 31% vs ~7% in reports), so the fitted
values sit high in absolute terms. Only their spacing matters for ranking; BCE on soft targets tolerates the offset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from knee.labels import prompt as P
from knee.paths import LABELS


def fit_map(bins: pd.Series, y: pd.Series, label: str, m: float = 5.0) -> dict[str, float]:
    """bins/y aligned on study; bins None = not mentioned (ignored here)."""
    vals, wts = [], []
    for b in P.BINS:
        sel = bins == b
        n, pos = float(sel.sum()), float(y[sel].sum())
        prior = P.prior_prob(label, b)
        vals.append((pos + m * prior) / (n + m))
        wts.append(n + m)
    iso = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0)
    fitted = iso.fit_transform(np.arange(len(P.BINS)), np.asarray(vals), sample_weight=np.asarray(wts))
    return dict(zip(P.BINS, map(float, fitted)))


def prior_map(label: str) -> dict[str, float]:
    return {b: P.prior_prob(label, b) for b in P.BINS}


def apply_map(bins: pd.Series, mp: dict[str, float]) -> pd.Series:
    return bins.map(lambda b: mp.get(b, np.nan) if isinstance(b, str) else np.nan).astype(float)


def reader_frame(long: pd.DataFrame, gold: pd.DataFrame | None, calibrate: bool = True, m: float = 5.0
                 ) -> tuple[pd.DataFrame, dict]:
    """Long frame from reader.parse_run → standard frame (NaN = not mentioned).

    calibrate=False → prompt.PRIOR values. calibrate=True → the gold map; gold studies get their leave-one-out value.
    Returns (frame, maps) where maps[label] is the all-gold map (for the QA report)."""
    wide_bin = long.pivot(index="StudyInstanceUID", columns="label", values="bin").reindex(columns=LABELS)
    out = pd.DataFrame(index=wide_bin.index, columns=LABELS, dtype=float)
    maps: dict[str, dict] = {}
    gidx = [] if gold is None else [u for u in gold.index if u in wide_bin.index]
    for k in LABELS:
        b = wide_bin[k].where(wide_bin[k].notna(), None)
        if not calibrate or not gidx:
            maps[k] = prior_map(k)
            out[k] = apply_map(b, maps[k])
            continue
        yk = gold.loc[gidx, k].astype(float)
        maps[k] = fit_map(b.loc[gidx], yk, k, m)
        out[k] = apply_map(b, maps[k])
        for u in gidx:  # leave-one-out for the gold rows
            rest = [v for v in gidx if v != u]
            mp = fit_map(b.loc[rest], yk.loc[rest], k, m)
            out.loc[u, k] = apply_map(b.loc[[u]], mp).iloc[0]
    out.index.name = "StudyInstanceUID"
    return out, maps
