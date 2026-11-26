"""Split and dividend adjustment of raw bars, as known at a decision time (D-030, doc 03 §3).

Raw bars stay raw in the DB. Adjusted prices exist only in memory: backward adjustment, so the
newest bar keeps its traded price and every earlier bar is scaled by the actions between it and T.

Point-in-time: at T only actions with ``released_at <= T`` and ``ex_date <= T``'s bar date count.
An action released later never touches history at T. (Adding a later action multiplies every bar
up to T by one constant, so scale-free features such as returns, vol and ratios come out the same
either way. Price levels and dollar volume differ, and features must not depend on them.)

Factors, per action, applied to bars dated before the ex date:
- split (forward/reverse/unit): price x old/new, volume x new/old
- cash dividend: price x (1 - D / raw close of the last bar before the ex date) (total return)
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from qt.common.config import AdjustCfg

log = logging.getLogger("qt.features")

SPLIT_KINDS = frozenset({"ca_forward_splits", "ca_reverse_splits", "ca_unit_splits"})
DIVIDEND_KINDS = frozenset({"ca_cash_dividends"})


def action_factors(
    bars: pd.DataFrame, actions: pd.DataFrame, asof: datetime, cfg: AdjustCfg
) -> pd.DataFrame:
    """One row per usable action of one asset: ``ex_date, price_f, volume_f``.

    ``bars`` holds one asset's raw bars (``date``, ``close``) known at ``asof``; ``actions`` that
    asset's mapped actions. Actions not yet released at ``asof`` are left out.
    """
    out = pd.DataFrame(columns=["ex_date", "price_f", "volume_f"])
    if actions.empty or bars.empty:
        return out
    t = pd.Timestamp(asof)
    last_bar = bars["date"].max()
    acts = actions[(actions["released_at"] <= t) & (actions["ex_date"] <= last_bar)]
    acts = acts[~acts["kind"].isin(cfg.ignore_kinds)]
    dates = pd.Index(bars["date"])
    closes = bars["close"].to_numpy()
    window = pd.Timedelta(days=cfg.split_bar_window_days)
    rows = []
    for a in acts.itertuples(index=False):
        manual = bool(getattr(a, "manual", False))
        if a.kind in SPLIT_KINDS and cfg.splits:
            if not (a.old_rate and a.new_rate) or a.old_rate <= 0 or a.new_rate <= 0:
                log.warning("split_bad_ratio", extra={"external_id": a.external_id})
                continue
            f = a.old_rate / a.new_rate
            if not manual and not _split_matches_prices(dates, closes, a.ex_date, f, window, cfg):
                log.warning(
                    "split_rejected_by_prices",
                    extra={"external_id": a.external_id, "ex_date": str(a.ex_date)},
                )
                continue
            rows.append((a.ex_date, f, 1.0 / f))
        elif a.kind in DIVIDEND_KINDS and cfg.dividends:
            if a.special and not cfg.special_dividends:
                continue
            pos = dates.searchsorted(a.ex_date) - 1
            if pos < 0 or not np.isfinite(a.rate) or a.rate <= 0:
                continue
            prior = closes[pos]
            frac = a.rate / prior if prior > 0 else np.inf
            if frac >= (1.0 if manual else cfg.max_dividend_frac):
                log.warning(
                    "dividend_too_large", extra={"external_id": a.external_id, "frac": frac}
                )
                continue
            rows.append((a.ex_date, 1.0 - frac, 1.0))
    if not rows:
        return out
    return pd.DataFrame(rows, columns=["ex_date", "price_f", "volume_f"])


def _split_matches_prices(
    dates: pd.Index,
    closes: np.ndarray,
    ex_date,
    factor: float,
    window: pd.Timedelta,
    cfg: AdjustCfg,
) -> bool:
    """True if the raw close move into ``ex_date`` agrees with the split ratio ``factor``."""
    after = dates.searchsorted(ex_date)  # first bar on/after the ex date
    before = after - 1
    if before < 0 or after >= len(dates):
        return False
    if (ex_date - dates[before]) > window or (dates[after] - ex_date) > window:
        return False
    if closes[before] <= 0 or closes[after] <= 0:
        return False
    actual = closes[after] / closes[before]
    return abs(np.log(actual / factor)) <= cfg.split_check_tol


def adjust(
    bars: pd.DataFrame, actions: pd.DataFrame, asof: datetime, cfg: AdjustCfg
) -> pd.DataFrame:
    """One asset's bars known at ``asof`` with ``adj_open/high/low/close`` and ``adj_volume``.

    Raw ``close`` and ``volume`` are kept: dollar volume uses them (what actually traded).
    """
    b = bars.sort_values("date").reset_index(drop=True)
    f = action_factors(b, actions, asof, cfg)
    price = np.ones(len(b))
    vol = np.ones(len(b))
    if not f.empty:
        d = pd.Index(b["date"])
        for r in f.itertuples(index=False):
            before = d < r.ex_date
            price[before] *= r.price_f
            vol[before] *= r.volume_f
    out = b.copy()
    for c in ("open", "high", "low", "close"):
        out[f"adj_{c}"] = out[c] * price
    out["adj_volume"] = out["volume"] * vol
    out["price_factor"] = price
    return out
