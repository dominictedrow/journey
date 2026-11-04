"""TSM02 run, exactly as pre-registered in research/TSM02_DESIGN.md (dc7f575). One shot.

Usage: uv run python scripts/tsm02_run.py
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trader import __version__
from trader.competition import git_sha

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / "data" / "yahoo"
RATES = REPO.parent / "data" / "rates"
ARTIFACTS = REPO / "artifacts"
HOLDOUT = pd.Timestamp("2025-10-01")
CLASSES = {
    "equities": ["SPY", "IWM", "EFA", "EWJ", "EWG", "EWU", "EEM"],
    "bonds": ["IEF", "TLT", "LQD", "TIP", "BWX"],
    "commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "DBB"],
    "currencies": ["FXE", "FXY", "FXB", "FXA", "FXC", "FXF"],
}
UNIVERSE = [t for ts in CLASSES.values() for t in ts]
TARGET = 0.40
COST = 0.0005  # one-way, per unit turnover
BORROW = 0.005  # per year on short notional
MIN_ASSETS = 12
VOL_COM = 60


def cash_daily() -> pd.Series:
    df = pd.read_csv(RATES / "DFF.csv", na_values=".")
    s = pd.Series(df.iloc[:, 1].to_numpy() / 100, index=pd.to_datetime(df.iloc[:, 0]))
    return s.reindex(pd.date_range(s.index.min(), "2026-12-31", freq="D")).ffill()


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly excess returns and month-end ex-ante annual vol, per asset."""
    dff = cash_daily()
    monthly, vols = {}, {}
    for t in UNIVERSE:
        d = pd.read_csv(DATA / f"{t}.csv", parse_dates=["date"]).set_index("date")["adjclose"]
        d = d[d.index < HOLDOUT]
        days = d.index.to_series().diff().dt.days
        rf = dff.reindex(d.index).shift(1).to_numpy() / 360 * days.to_numpy()
        ex = (d.pct_change() - rf).dropna()
        vol = ex.ewm(com=VOL_COM, min_periods=VOL_COM).std() * math.sqrt(261)
        monthly[t] = (1 + ex).groupby(ex.index.to_period("M")).prod() - 1
        vols[t] = vol.groupby(vol.index.to_period("M")).last()
    return pd.DataFrame(monthly), pd.DataFrame(vols)


def signals(mret: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Sign of trailing cumulative excess return over `lookback` complete months (known at t)."""
    cum = np.log1p(mret).rolling(lookback, min_periods=lookback).sum()
    return np.sign(cum)


def portfolio(mret, vol, lookback, cost_x=1.0, long_flat=False, assets=None):
    assets = assets or UNIVERSE
    sig = signals(mret[assets], lookback)
    if long_flat:
        sig = sig.clip(lower=0)
    w = (sig * TARGET / vol[assets]).where(vol[assets] > 0)
    live = w.notna()
    # Position decided at month-end t earns month t+1.
    w_prev = w.shift(1)
    nxt = mret[assets]
    pnl = (w_prev * nxt).where(live.shift(1, fill_value=False))
    was_live = live.shift(1, fill_value=False)
    turnover = (w_prev.fillna(0) - w_prev.shift(1).fillna(0)).abs().where(was_live)
    borrow = (-w_prev.clip(upper=0)) * BORROW / 12
    net = pnl - turnover * COST * cost_x - borrow * cost_x
    n_live = live.shift(1, fill_value=False).sum(axis=1)
    port = net.sum(axis=1, min_count=1) / n_live.where(n_live > 0)
    return port, n_live


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * math.sqrt(12)) if len(r) > 2 and r.std() > 0 else 0.0


def eras(r: pd.Series) -> dict[str, tuple[float, float]]:
    out = {}
    idx = r.index.to_timestamp()
    for name, a, b in [("start-2012", "1900", "2013"), ("2013-2019", "2013", "2020"),
                       ("2020-2025", "2020", "2030")]:  # fmt: skip
        s = r[(idx >= a) & (idx < b)]
        out[name] = (sharpe(s), float((1 + s).prod() - 1))
    return out


def main() -> None:
    mret, vol = load_panel()
    base, n_live = portfolio(mret, vol, 12)
    start = n_live[n_live >= MIN_ASSETS].index.min()
    print(f"universe {len(UNIVERSE)} ETFs; dev window {start} -> 2025-09 "
          f"(first month with >= {MIN_ASSETS} assets); "
          f"holdout sealed from {HOLDOUT.date()}")  # fmt: skip
    print(f"assets live: min {n_live[start:].min()}, median {n_live[start:].median():.0f}")

    def window(r):
        return r[r.index >= start]

    spy = window(mret["SPY"])
    rows = {}
    variants = {
        "PRIMARY L=12 long/short": dict(lookback=12),
        "  L=12 long/short, 3x costs": dict(lookback=12, cost_x=3.0),
        "secondary L=3 long/short": dict(lookback=3),
        "secondary L=12 long/flat": dict(lookback=12, long_flat=True),
        **{f"secondary L=12 {k} only": dict(lookback=12, assets=v)
           for k, v in CLASSES.items()},  # fmt: skip
    }
    for label, kw in variants.items():
        r = window(portfolio(mret, vol, **kw)[0])
        vol_ann = r.std() * math.sqrt(12)
        scaled = r.mean() * 12 * (0.10 / vol_ann) if vol_ann > 0 else 0.0
        e = eras(r)
        rows[label] = r
        corr = r.corr(spy)
        print(f"{label:<32} Sharpe {sharpe(r):+.2f}  ann.ret {r.mean() * 12:+.1%} (vol "
              f"{vol_ann:.1%}; at 10% vol {scaled:+.1%})  corrSPY {corr:+.2f}  | "
              + "  ".join(f"{k} {s:+.2f}/{tot:+.0%}" for k, (s, tot) in e.items()))  # fmt: skip

    print(f"\ncontext: SPY excess Sharpe {sharpe(spy):+.2f}; equal-weight buy-and-hold of the "
          f"universe Sharpe {sharpe(window(mret.mean(axis=1))):+.2f}")  # fmt: skip

    r1, r3 = rows["PRIMARY L=12 long/short"], rows["  L=12 long/short, 3x costs"]
    e1 = eras(r1)
    checks = {
        "1. Sharpe > 0.40": sharpe(r1) > 0.40,
        "2. positive every era": all(tot > 0 for _, tot in e1.values()),
        "3. Sharpe > 0.20 at 3x": sharpe(r3) > 0.20,
        "4. corr with SPY < 0.30": r1.corr(spy) < 0.30,
    }
    v = "PASS" if all(checks.values()) else "FAIL"
    print(f"\npre-registered verdict: {v}  "
          + "; ".join(f"{k}={'ok' if ok else 'NO'}" for k, ok in checks.items()))  # fmt: skip

    out = ARTIFACTS / "tsm02"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "monthly_returns.csv")
    rec = {
        "time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code": f"{__version__}@{git_sha(REPO)}", "label": "tsm02", "holdout_unsealed": False,
        "start": str(start), "end": "2025-09",
        "metrics": {k: {"sharpe": sharpe(r), "corr_spy": float(r.corr(spy))}
                    for k, r in rows.items()},
        "verdict": v,
    }  # fmt: skip
    with (ARTIFACTS / "experiments.jsonl").open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


if __name__ == "__main__":
    main()
