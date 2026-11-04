"""PORT01 run, exactly as pre-registered in research/PORT01_DESIGN.md (cde5247). One shot.

Sleeve signals and vols come from scripts/tsm02_run.py unchanged (TSM02 primary).

Usage: uv run python scripts/port01_run.py
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import tsm02_run as tsm  # same directory; TSM02 primary signal code, unchanged

from trader import __version__
from trader.competition import git_sha

CORE = 0.60
SLEEVE_M = 0.5
GROSS_CAP = 2.0
MARGIN_SPREAD = 0.015  # per year over DFF on borrowed cash
BORROW = 0.005  # per year on short notional
COST = 0.0005  # one-way per unit turnover


def total_monthly() -> pd.DataFrame:
    out = {}
    for t in tsm.UNIVERSE:
        d = pd.read_csv(tsm.DATA / f"{t}.csv", parse_dates=["date"]).set_index("date")["adjclose"]
        d = d[d.index < tsm.HOLDOUT]
        r = d.pct_change().dropna()
        out[t] = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    return pd.DataFrame(out)


def cash_monthly() -> pd.Series:
    dff = tsm.cash_daily()
    daily = 1 + dff / 360
    return daily.groupby(daily.index.to_period("M")).prod() - 1


def sleeve_weights(mex: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    sig = tsm.signals(mex[tsm.UNIVERSE], 12)
    w = (sig * tsm.TARGET / vol[tsm.UNIVERSE]).where(vol[tsm.UNIVERSE] > 0)
    n_live = w.notna().sum(axis=1)
    return w.div(n_live.where(n_live > 0), axis=0).fillna(0.0) * SLEEVE_M


def combine(sleeve: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Core + sleeve with SPY netted; sleeve scaled down so gross <= cap. Returns weights, f."""
    rows, fs = {}, {}
    core = pd.Series(0.0, index=sleeve.columns)
    core["SPY"] = CORE
    for t, s in sleeve.iterrows():

        def gross(f, s=s):
            return float((core + f * s).abs().sum())

        f = 1.0
        if gross(1.0) > GROSS_CAP:
            lo, hi = 0.0, 1.0
            for _ in range(50):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if gross(mid) <= GROSS_CAP else (lo, mid)
            f = lo
        rows[t] = core + f * s
        fs[t] = f
    return pd.DataFrame(rows).T, pd.Series(fs)


def run(W: pd.DataFrame, R: pd.DataFrame, rf: pd.Series, mult: float = 1.0) -> pd.Series:
    """Monthly total return: weights decided at t earn month t+1 (retail margin accounting)."""
    W = W.reindex(columns=R.columns, fill_value=0.0)
    Wp = W.shift(1)
    Wpp = W.shift(2).fillna(0.0)
    rf_n = rf.reindex(R.index)
    out = {}
    for t in R.index:
        if t not in Wp.index or Wp.loc[t].isna().all():
            continue
        w = Wp.loc[t].fillna(0.0)
        r = R.loc[t].fillna(0.0)
        long_sum = w.clip(lower=0).sum()
        cash = 1 - long_sum
        rf_t = rf_n.loc[t]
        cash_ret = cash * rf_t if cash >= 0 else cash * (rf_t + MARGIN_SPREAD * mult / 12)
        borrow = (-w.clip(upper=0)).sum() * BORROW * mult / 12
        turnover = (w - Wpp.loc[t]).abs().sum()
        out[t] = float((w * r).sum() + cash_ret - borrow - turnover * COST * mult)
    return pd.Series(out)


def stats(r: pd.Series, rf: pd.Series) -> dict[str, float]:
    ex = r - rf.reindex(r.index)
    eq = (1 + r).cumprod()
    dd = float((1 - eq / eq.cummax()).max())
    years = len(r) / 12
    return {
        "sharpe": float(ex.mean() / ex.std() * math.sqrt(12)) if ex.std() > 0 else 0.0,
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
        "vol": float(r.std() * math.sqrt(12)),
        "max_dd": dd,
    }


ERAS = [
    ("start-2012", "1900", "2013"),
    ("2013-2019", "2013", "2020"),
    ("2020-2025", "2020", "2030"),
]


def era_sharpes(r: pd.Series, rf: pd.Series) -> dict[str, float]:
    idx = r.index.to_timestamp()
    return {n: stats(r[(idx >= a) & (idx < b)], rf)["sharpe"] for n, a, b in ERAS}


def main() -> None:
    mex, vol = tsm.load_panel()
    R = total_monthly()
    rf = cash_monthly()
    sleeve = sleeve_weights(mex, vol)
    _, n_live = tsm.portfolio(mex, vol, 12)  # TSM02's own window definition
    start = n_live[n_live >= tsm.MIN_ASSETS].index.min()
    W, f = combine(sleeve)
    print(f"PORT01 dev window {start} -> 2025-09; holdout sealed from {tsm.HOLDOUT.date()}")
    gross = W.abs().sum(axis=1)[start:]
    print(f"gross exposure: mean {gross.mean():.2f}, max {gross.max():.2f}; "
          f"cap binding in {(f[start:] < 1).mean():.0%} of months")  # fmt: skip

    def bench(weights: dict[str, float]) -> pd.DataFrame:
        b = pd.DataFrame(0.0, index=W.index, columns=R.columns)
        for k, v in weights.items():
            b[k] = v
        return b

    cases = {
        "PORT01": (W, 1.0),
        "PORT01 3x costs": (W, 3.0),
        "B1 100% SPY": (bench({"SPY": 1.0}), 1.0),
        "B2 60% SPY / 40% cash": (bench({"SPY": 0.6}), 1.0),
        "B3 60% SPY / 40% IEF": (bench({"SPY": 0.6, "IEF": 0.4}), 1.0),
    }
    res, out = {}, {}
    for name, (w, mult) in cases.items():
        r = run(w, R, rf, mult)
        r = r[r.index >= start]
        out[name] = r
        s = stats(r, rf)
        e = era_sharpes(r, rf)
        res[name] = (s, e)
        eras = "  ".join(f"{k} {v:+.2f}" for k, v in e.items())
        print(f"{name:<24} Sharpe {s['sharpe']:+.2f}  CAGR {s['cagr']:+.1%}  "
              f"vol {s['vol']:.1%}  maxDD {s['max_dd']:.1%}  | {eras}")  # fmt: skip

    p, p3, b1, b3 = (res[k] for k in ("PORT01", "PORT01 3x costs", "B1 100% SPY",
                                       "B3 60% SPY / 40% IEF"))  # fmt: skip
    checks = {
        "1. Sharpe >= SPY + 0.10": p[0]["sharpe"] >= b1[0]["sharpe"] + 0.10,
        "2. maxDD <= 0.75 x SPY": p[0]["max_dd"] <= 0.75 * b1[0]["max_dd"],
        "3. Sharpe >= 60/40": p[0]["sharpe"] >= b3[0]["sharpe"],
        "4. 3x Sharpe >= SPY": p3[0]["sharpe"] >= b1[0]["sharpe"],
        "5. every era >= SPY - 0.10": all(p[1][k] >= b1[1][k] - 0.10 for k in p[1]),
    }
    v = "PASS" if all(checks.values()) else "FAIL"
    print(f"\npre-registered verdict: {v}  "
          + "; ".join(f"{k}={'ok' if ok else 'NO'}" for k, ok in checks.items()))  # fmt: skip

    d = tsm.ARTIFACTS / "port01"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(d / "monthly_returns.csv")
    W.to_csv(d / "weights.csv")
    rec = {
        "time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code": f"{__version__}@{git_sha(tsm.REPO)}", "label": "port01",
        "holdout_unsealed": False, "start": str(start), "end": "2025-09",
        "metrics": {k: s for k, (s, _) in res.items()}, "verdict": v,
    }  # fmt: skip
    with (tsm.ARTIFACTS / "experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
