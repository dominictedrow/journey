"""The metric code every track shares (02 §B, master plan §6).

Competition metric [Kaggle]: macro-averaged AUC-ROC over the 12 findings. A finding whose targets are all one class
(common on small folds / the gold 58) has no AUC; it is skipped and reported as NaN, and `macro_auc` says how many
findings it averaged. Soft report labels are binarized for AUC with a dead band: ≥ hi → 1, ≤ lo → 0, in between →
excluded (HP: lo, hi).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee.paths import LABELS


def binarize(y: np.ndarray, lo: float = 0.4, hi: float = 0.6) -> np.ndarray:
    """Soft targets → {0, 1, NaN}. NaN in stays NaN."""
    y = np.asarray(y, dtype=np.float64)
    out = np.full_like(y, np.nan)
    out[y >= hi] = 1.0
    out[y <= lo] = 0.0
    return out


def per_finding_auc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """y (n, 12) in {0, 1, NaN}, p (n, 12) → (12,) AUCs, NaN where undefined."""
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    out = np.full(y.shape[1], np.nan)
    for k in range(y.shape[1]):
        ok = np.isfinite(y[:, k]) & np.isfinite(p[:, k])
        yk = y[ok, k]
        if ok.sum() >= 2 and 0 < yk.sum() < len(yk):
            out[k] = roc_auc_score(yk, p[ok, k])
    return out


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    a = per_finding_auc(y, p)
    n = int(np.isfinite(a).sum())
    return (float(np.nanmean(a)) if n else float("nan")), n


def bootstrap_macro_auc(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = 0,
                        alpha: float = 0.05) -> dict:
    """Resample studies with replacement; macro AUC per replicate. Returns point, lo, hi (percentile CI)."""
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    rng = np.random.default_rng(seed)
    point, _ = macro_auc(y, p)
    reps = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        v, n = macro_auc(y[idx], p[idx])
        if n:
            reps.append(v)
    reps = np.asarray(reps)
    return {"auc": point, "lo": float(np.quantile(reps, alpha / 2)) if len(reps) else float("nan"),
            "hi": float(np.quantile(reps, 1 - alpha / 2)) if len(reps) else float("nan"), "n_boot": len(reps)}


def rank_normalize(p: np.ndarray) -> np.ndarray:
    """Per column: average ranks scaled to (0, 1]. AUC reads only order, so blends are done on ranks [KT]."""
    return pd.DataFrame(p).rank(axis=0, method="average", pct=True).to_numpy()


def rank_blend(preds: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    w = np.ones(len(preds)) if weights is None else np.asarray(weights, dtype=np.float64)
    return sum(wi * rank_normalize(pi) for wi, pi in zip(w, preds)) / w.sum()


def auc_table(y: np.ndarray, p: np.ndarray, name: str = "auc") -> pd.DataFrame:
    """One row per finding + a MACRO row, with positives/negatives counted (for docs/experiments.md)."""
    y = np.asarray(y, dtype=np.float64)
    a = per_finding_auc(y, p)
    df = pd.DataFrame({"finding": LABELS, name: a, "n_pos": np.nansum(y == 1, axis=0).astype(int),
                       "n_neg": np.nansum(y == 0, axis=0).astype(int)})
    macro, n = macro_auc(y, p)
    return pd.concat([df, pd.DataFrame([{"finding": f"MACRO ({n})", name: macro, "n_pos": -1, "n_neg": -1}])],
                     ignore_index=True)


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """A pipe table without the optional `tabulate` dependency."""
    def cell(v):
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"
