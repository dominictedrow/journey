"""Normalization, masks and staleness (component doc 03 section 7).

- **Time-series z** (``<name>_tsz<w>``): the feature's newest value against its own last ``w``
  values (this one included; past only), winsorized at +-``winsor``. Both the raw feature and its
  z are stored. Market features: over the market vector's values at the last ``w`` decision times
  (sessions). Per-asset features of unranked classes (ETFs, coins): over the asset's own bar rows.
  A z needs ``min_frac`` of its window finite (data gaps) and its own value finite.
- **Masks** (Q8, per asset): ``history_days`` (bars so far) and ``hist_ok_<n>`` (1 once the
  asset has ``n`` returns, i.e. ``n + 1`` bars).
- **Staleness** (section 3): ``dsr_<series>`` = calendar days from when the newest period of a
  macro series was first released to T. A revision of an older period doesn't reset it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from qt.common.config import NormalizeCfg
from qt.features import registry
from qt.features.market import MarketState

G_TSZ_MARKET = "market_ts_z"
G_TSZ_ASSET = "asset_ts_z"
G_MASKS = "masks"
G_STALENESS = "macro_staleness"
NS_PER_DAY = 86_400 * 10**9


# ---------------------------------------------------------------- z
def rolling_max_int(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing max over ``window`` rows (fewer at the start), exact in int64.

    ``Series.rolling`` goes through float64, which rounds epoch-ns timestamps and turns the
    int64-max "never ready" sentinel into int64-min on the way back.
    """
    x = np.asarray(x, dtype="int64")
    if x.size == 0:
        return x.copy()
    pad = np.concatenate([np.full(window - 1, np.iinfo(np.int64).min, dtype="int64"), x])
    return np.lib.stride_tricks.sliding_window_view(pad, window).max(axis=1)


def rolling_z(x: np.ndarray, window: int, winsor: float, min_frac: float) -> np.ndarray:
    """z of each value against the last ``window`` values (itself included), clipped."""
    s = pd.Series(np.asarray(x, dtype=float))
    need = max(math.ceil(min_frac * window), 3)
    r = s.rolling(window, min_periods=need)
    mean, sd = r.mean(), r.std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ((s - mean) / sd).where(sd > 0)
    return z.clip(-winsor, winsor).to_numpy()


def z_last(hist: np.ndarray, window: int, winsor: float, min_frac: float) -> float:
    """:func:`rolling_z` of the newest value of ``hist`` (only its last ``window`` matter)."""
    h = np.asarray(hist, dtype=float)[-window:]
    if not len(h):
        return math.nan
    v = float(rolling_z(h, window, winsor, min_frac)[-1])
    return v if math.isfinite(v) else math.nan


def tsz_name(base: str, window: int) -> str:
    return f"{base}_tsz{window}"


def register_ts_z(
    base: list[registry.FeatureSpec], cfg: NormalizeCfg, groups: list[str]
) -> list[registry.FeatureSpec]:
    """A ``_tsz`` feature for every spec of ``base`` in ``groups`` (same kind as its base).

    ``fn`` takes the base feature's value history (oldest first) and returns the z of each row
    (asset kind: the whole array) or of the newest row (market kind: a float). A z's version is
    its base's version, so bumping a base bumps its z (older base versions keep their old z).
    """
    keys = []
    w, cap, frac = cfg.z_window, cfg.winsor, cfg.min_frac
    for b in base:
        if b.group not in groups:
            continue
        name = tsz_name(b.name, w)
        group = G_TSZ_ASSET if b.kind == "asset" else G_TSZ_MARKET
        fn: Callable
        if b.kind == "asset":
            def fn(x, w=w, cap=cap, frac=frac):
                return rolling_z(x, w, cap, frac)
        else:
            def fn(x, w=w, cap=cap, frac=frac):
                return z_last(x, w, cap, frac)
        registry.feature(name, b.version, kind=b.kind, group=group, window=w,
                         formula=f"z of {b.name} v{b.version} vs its last {w} values (incl. "
                                 f"this one), clipped to +-{cap:g}, needs {frac:g} of them",
                         sources=(f"feature:{b.name}:v{b.version}",))(fn)  # fmt: skip
        keys.append((name, b.version))
    return [registry.get(*k) for k in keys]


def base_key(spec: registry.FeatureSpec) -> tuple[str, int]:
    """The ``(name, version)`` a ``_tsz`` spec normalizes."""
    _, name, ver = spec.sources[0].split(":")
    return (name, int(ver[1:]))


def asset_ts_z(
    values: pd.DataFrame, ready: pd.DataFrame, specs: list[registry.FeatureSpec]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asset ``_tsz`` values and ready times from a :func:`qt.features.price.compute_series` pair.

    A z row may be used once every finite base value in its window is ready.
    """
    vals, rdy = {}, {}
    for s in specs:
        b = base_key(s)
        x = values[b].to_numpy(dtype=float)
        r = np.where(np.isfinite(x), ready[b].to_numpy(), 0)
        w = int(s.window)
        vals[s.key] = s.fn(x)
        rdy[s.key] = np.maximum(
            rolling_max_int(r, w),
            ready[b].to_numpy(dtype="int64"),
        )
    cols = pd.Index([s.key for s in specs], tupleize_cols=False)
    v = pd.DataFrame(vals, index=values.index)
    r = pd.DataFrame(rdy, index=values.index)
    v.columns, r.columns = cols, cols
    return v, r


# ---------------------------------------------------------------- masks (asset)
def register_masks(cfg: NormalizeCfg) -> list[registry.FeatureSpec]:
    keys = []

    def reg(name, formula, window, fn):
        registry.feature(name, 1, kind="asset", group=G_MASKS, formula=formula, window=window,
                         sources=("bars",))(fn)  # fmt: skip
        keys.append((name, 1))

    reg("history_days", "number of bars up to and including this one", None,
        lambda f: np.arange(1, f.n + 1, dtype=float))  # fmt: skip
    for n in cfg.mask_windows:
        reg(f"hist_ok_{n}", f"1 if the asset has {n} returns ({n + 1} bars) so far, else 0", n,
            lambda f, n=n: (np.arange(1, f.n + 1) >= n + 1).astype(float))  # fmt: skip
    return [registry.get(*k) for k in keys]


# ---------------------------------------------------------------- staleness (market)
def macro_sources(specs: list[registry.FeatureSpec], groups: list[str]) -> list[str]:
    """FRED series read by the specs in ``groups`` (prices ``px:``, breadth etc. excluded)."""
    out = set()
    for s in specs:
        if s.group in groups:
            out.update(x for x in s.sources if ":" not in x and x.isupper())
    return sorted(out)


def days_since_release(m: MarketState, sid: str) -> float:
    top = m.macro.newest_period_release(sid)
    if top is None:
        return math.nan
    return (m.asof_ns - int(top[1].as_unit("ns").value)) / NS_PER_DAY


def register_staleness(
    specs: list[registry.FeatureSpec], cfg: NormalizeCfg
) -> list[registry.FeatureSpec]:
    keys = []
    for sid in macro_sources(specs, cfg.staleness_groups):
        name = f"dsr_{sid.lower()}"
        registry.feature(name, 1, kind="market", group=G_STALENESS, sources=(sid,),
                         formula=f"days from the first release of {sid}'s newest period to T")(
            lambda m, sid=sid: days_since_release(m, sid))  # fmt: skip
        keys.append((name, 1))
    return [registry.get(*k) for k in keys]
