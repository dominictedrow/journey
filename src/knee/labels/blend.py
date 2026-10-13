"""Readers → one soft label p and one confidence weight w per (study, finding) (S04 step 6).

Inputs: standard frames (public_sets.py docstring): index = StudyInstanceUID, 12 LABELS columns, NaN = silent.

Per cell:
1. **Who spoke.** A silence-aware reader spoke when it isn't NaN. Readers without a silence signal (`NO_SILENCE`,
   e.g. lixin, which writes 0 when the report is silent) "spoke" only when they assert a positive (≥ 0.5): a lone 0
   from them is silence in disguise. A cell is *known* when anyone spoke, else *silent*.
2. **Consensus** p̄ on known cells, in probability space (not rank space: `2|p−0.5|` must keep its meaning
   [ds:stevenleehans]). Equal reader weights; none are tuned on gold. Two ways to treat a silent reader in a known
   cell (`silence`):
   - `"vote"` (default): every reader votes; a silent silence-aware reader votes `silent_p` = 0.5 ("unknown"), which is
     how each public set scores silence on its own. So one explicit negative among three silent readers gives ~0.4,
     not ~0.05.
   - `"skip"`: mean over the readers with a value only (a lone negative wins outright; in a known cell a
     non-aware reader's 0 counts, as in "vote").
   Gold 58 (public readers only, S04): vote 0.887 vs skip 0.883 macro, Fracture 0.816 vs 0.779, Lateral OA 0.821 vs
   0.793. Within the ±0.04 gold CI; "vote" is the default because it matches the readers' own convention, not
   because of that number.
3. **Weight** (prvsiyan `build_public_consensus`, public and measured):
   agreement = 1 − 2·mean|p_i − p̄|, certainty = 2|p̄ − 0.5|, w = 0.20 + 0.80·(0.65·agreement + 0.35·certainty).
   `weight_mode="certainty"` gives w = 2|p̄ − 0.5| (the brief's first option, an ablation).
4. **Silent cells** (nobody spoke): co-occurrence fills from the blended source finding when a rule exists
   (Synovitis ← Effusion, PF OA ← Medial OA), else `silent_p`. Both get a low weight.
   Fill constants [KL]: P(gold synovitis | effusion) = 0.63 vs 0.22 without [ds:stevenleehans]; PF OA ← Medial OA at
   0.60 [nb:laymond]. They were read off public gold analyses, so they are gold-informed but not tuned by us.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from knee.paths import LABELS

NO_SILENCE = {"lixin", "jev_final2", "jev_hybrid", "steven_v4"}


@dataclass
class BlendConfig:
    weight_mode: str = "agree_cert"         # "agree_cert" | "certainty" | "ones"
    silent_p: float = 0.50                  # "unknown": the readers' own convention (steven 0.5, laymond/pilkwang UNK)
    silence: str = "vote"                   # "vote" | "skip" (see module docstring, step 2)
    silent_w: float = 0.10
    # Silent cells (nobody spoke, no fill) only. None = silent_p (v0). E016: on gold, 4 of 99 silent cells are positive,
    # so v1 treats silence as a soft negative: p = silent_cell_p at weight silent_cell_w (HP placeholders).
    silent_cell_p: float | None = None
    silent_cell_w: float = 0.50
    fill_w: float = 0.30
    # target ← source: (p when source=0, p when source=1), linear in the blended source p. Placeholders [KL].
    fills: dict = field(default_factory=lambda: {"Synovitis": ("Effusion", 0.22, 0.63),
                                                 "PF OA": ("Medial OA", 0.20, 0.60)})
    reader_weights: dict = field(default_factory=dict)  # name → weight (default 1); not tuned on gold


def blend(readers: dict[str, pd.DataFrame], uids: list[str], cfg: BlendConfig | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (p, w, status). status per cell: 'known' | 'fill' | 'silent'. Studies a reader lacks count as silent
    for that reader."""
    cfg = cfg or BlendConfig()
    names = list(readers)
    R = np.stack([readers[n].reindex(uids)[LABELS].to_numpy(np.float64) for n in names])  # (r, n, 12)
    aware = np.array([n not in NO_SILENCE for n in names])
    rw = np.array([float(cfg.reader_weights.get(n, 1.0)) for n in names])[:, None, None]

    finite = np.isfinite(R)
    spoke = finite & (aware[:, None, None] | (R >= 0.5))
    known = spoke.any(axis=0)
    if cfg.silence == "vote":  # every reader votes; silent aware readers vote silent_p, non-aware vote their value
        use = np.broadcast_to(aware[:, None, None], R.shape) | finite
        V = np.where(finite, R, cfg.silent_p)
    elif cfg.silence == "skip":
        use, V = finite & (aware[:, None, None] | known[None]), np.where(finite, R, 0.0)
    else:
        raise ValueError(cfg.silence)
    wsum = (use * rw).sum(axis=0)
    p = np.where(known, (np.where(use, V, 0.0) * rw).sum(axis=0) / np.maximum(wsum, 1e-12), np.nan)
    R = V

    dev = np.where(use, np.abs(R - p[None]), 0.0)
    agreement = 1.0 - 2.0 * (dev * rw).sum(axis=0) / np.maximum(wsum, 1e-12)
    certainty = 2.0 * np.abs(p - 0.5)
    if cfg.weight_mode == "agree_cert":
        w = 0.20 + 0.80 * (0.65 * agreement + 0.35 * certainty)
    elif cfg.weight_mode == "certainty":
        w = certainty
    elif cfg.weight_mode == "ones":
        w = np.ones_like(p)
    else:
        raise ValueError(cfg.weight_mode)

    status = np.where(known, "known", "silent").astype(object)
    col = {k: i for i, k in enumerate(LABELS)}
    for tgt, (src, lo, hi) in cfg.fills.items():
        t, s = col[tgt], col[src]
        ps = np.where(np.isfinite(p[:, s]), p[:, s], cfg.silent_p)
        m = ~known[:, t]
        p[m, t] = lo + (hi - lo) * ps[m]
        w[m, t] = cfg.fill_w
        status[m, t] = "fill"
    still = ~np.isfinite(p)
    neg = cfg.silent_cell_p is not None
    p[still] = cfg.silent_cell_p if neg else cfg.silent_p
    w[still] = cfg.silent_cell_w if neg else cfg.silent_w

    mk = lambda a: pd.DataFrame(a, index=pd.Index(uids, name="StudyInstanceUID"), columns=LABELS)  # noqa: E731
    return mk(np.clip(p, 0.0, 1.0)), mk(np.clip(w, 0.0, 1.0)), mk(status)


def reader_for_scoring(frame: pd.DataFrame, uids: list[str], silent_value: float = 0.5) -> pd.DataFrame:
    """A single reader as a full score matrix (silent → silent_value, missing study → silent_value)."""
    return frame.reindex(uids)[LABELS].fillna(silent_value)
