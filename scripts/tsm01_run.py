"""TSM01 run, exactly as pre-registered in research/experiments/TSM01_DESIGN.md (3b3bf52). One shot.

A. Scale-free monthly benchmark (both instruments, carry + costs).
B. Bot run on EUR/USD through Backtester under the PROPOSED risk rules, carry added per trade.

Usage: uv run python scripts/tsm01_run.py
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trader import __version__
from trader.backtest.engine import Backtester
from trader.backtest.metrics import compute_metrics
from trader.competition import CompetitionConfig, git_sha
from trader.data.loaders import load_bars
from trader.data.resample import resample_bars
from trader.instruments import get_instrument
from trader.modes import build_mode
from trader.modes.tsm import session_dates

REPO = Path(__file__).resolve().parents[1]
RATES = REPO.parent / "data" / "rates"
MARKUP = 0.01  # retail FX carry markup, per year, against both sides (ASSUMPTION)
DIV_YIELD = 0.017  # S&P dividend yield (ASSUMPTION)
ONE_WAY_COST = {"EURUSD": 0.00007, "USA500IDXUSD": 0.5}  # half of 1.4 pip / 1 MES point
EVAL_START = {"EURUSD": "2004-01-01", "USA500IDXUSD": "2012-01-16"}
LOAD_START = {"EURUSD": None, "USA500IDXUSD": "2012-01-16"}
ERAS = {
    "EURUSD": [("2004-2011", "2004", "2012"), ("2012-2019", "2012", "2020"),
               ("2020 on", "2020", "2030")],
    "USA500IDXUSD": [("2013-2019", "2012", "2020"), ("2020 on", "2020", "2030")],
}  # fmt: skip


def fred(series: str) -> pd.Series:
    df = pd.read_csv(RATES / f"{series}.csv", na_values=".")
    s = pd.Series(df.iloc[:, 1].to_numpy() / 100, index=pd.to_datetime(df.iloc[:, 0]))
    days = pd.date_range(s.index.min(), "2026-12-31", freq="D")
    return s.reindex(days).ffill()


def daily_carry(symbol: str, mult: float) -> tuple[pd.Series, float]:
    """Per-calendar-day carry for a LONG of 1 unit notional, and the per-day markup (both sides)."""
    usd = fred("DFF")
    if symbol == "EURUSD":
        eur = fred("ECBDFR").reindex(usd.index).ffill()
        return (eur - usd) / 365, MARKUP * mult / 365
    return (DIV_YIELD - usd) / 365, 0.0


def carry_between(cum: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> float:
    """Sum of daily values over calendar days (a, b]."""
    a, b = a.tz_localize(None).normalize(), b.tz_localize(None).normalize()
    return float(cum.asof(b) - cum.asof(a))


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * math.sqrt(12)) if len(r) > 2 and r.std() > 0 else 0.0


def benchmark(symbol: str, daily: pd.DataFrame, lookback: int, mult: float) -> pd.Series:
    mode = build_mode("tsm01", lookback_months=lookback)
    mode.prepare(daily)
    dec = np.flatnonzero(mode.feat["decide"])
    close = daily["close"].to_numpy()
    dates = session_dates(daily.index)
    sig = np.array([mode.signal(i).value for i in dec], float)
    sd_ann = mode.feat["sigma_m"][dec] / math.sqrt(21) * math.sqrt(252)
    w = np.where(np.isfinite(sd_ann) & (sd_ann > 0), sig * 0.10 / sd_ann, 0.0)
    w = np.nan_to_num(w)
    carry, markup = daily_carry(symbol, mult)
    cum = carry.cumsum()
    cost = ONE_WAY_COST[symbol] * mult
    out, prev_w = {}, 0.0
    for j in range(len(dec) - 1):
        i0, i1 = dec[j], dec[j + 1]
        d0, d1 = dates[i0], dates[i1]
        price_ret = close[i1] / close[i0] - 1
        c = carry_between(cum, pd.Timestamp(d0, tz="UTC"), pd.Timestamp(d1, tz="UTC"))
        ndays = (d1 - d0).days
        r = w[j] * price_ret + w[j] * c - abs(w[j]) * markup * ndays
        r -= abs(w[j] - prev_w) * cost / close[i0]
        prev_w = w[j]
        out[d1] = r
    s = pd.Series(out)
    return s[s.index >= EVAL_START[symbol]]


def buy_and_hold(symbol: str, daily: pd.DataFrame, months: pd.DatetimeIndex) -> pd.Series:
    close = daily["close"]
    close.index = session_dates(daily.index)
    px = close.reindex(months, method="ffill")
    carry, _ = daily_carry(symbol, 1.0)
    cum = carry.cumsum()
    r = {}
    for a, b in zip(px.index[:-1], px.index[1:], strict=True):
        c = carry_between(cum, pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"))
        r[b] = px[b] / px[a] - 1 + c
    return pd.Series(r)


def bot_run(cfg, bars, lookback, equity, mult, sha):
    inst = get_instrument("EURUSD")
    res = Backtester(inst, cfg.risk, equity, cost_multiplier=mult).run(
        build_mode("tsm01", lookback_months=lookback), bars
    )
    t = res.trades.copy()
    carry, markup = daily_carry("EURUSD", mult)
    cum = carry.cumsum()
    adj = []
    for row in t.itertuples():
        notional = row.units * row.entry_price  # USD
        c = carry_between(cum, row.entry_time, row.exit_time) * row.side
        ndays = (row.exit_time.normalize() - row.entry_time.normalize()).days
        adj.append(notional * (c - markup * ndays))
    t["carry"] = adj
    risk_amt = (t.entry_price - t.initial_stop).abs() * t.units
    t["pnl"] = t.pnl + t.carry
    t["r_multiple"] = t.pnl / risk_amt
    eq = res.daily_equity.copy()
    if len(t):
        exit_day = t.exit_time.dt.tz_localize(None).dt.normalize()
        realized = pd.Series(t.carry.to_numpy(), index=exit_day).groupby(level=0).sum()
        eq = eq + realized.reindex(eq.index, fill_value=0.0).cumsum()
    m = compute_metrics(t, eq, equity)
    m["carry_total"] = float(t.carry.sum()) if len(t) else 0.0
    rec = {
        "time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "code": f"{__version__}@{sha}",
        "symbol": "EURUSD", "start": str(bars.index[0]), "end": str(bars.index[-1]),
        "holdout_unsealed": False, "label": f"tsm01_L{lookback}:bot_{equity:g}",
        "mode": {"lookback_months": lookback}, "cost_x": mult, "risk": vars(cfg.risk),
        "metrics": m,
    }  # fmt: skip
    with (cfg.artifact_dir / "experiments.jsonl").open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return m, t


def main() -> None:
    cfg = CompetitionConfig.from_toml(REPO / "configs" / "competition.toml")
    sealed = json.loads((cfg.report_dir / "holdout.json").read_text())
    sha = git_sha(REPO)
    out_dir = cfg.artifact_dir / "tsm01"
    out_dir.mkdir(parents=True, exist_ok=True)
    verdicts = {}
    for symbol in ("EURUSD", "USA500IDXUSD"):
        end = pd.Timestamp(sealed[symbol]["holdout_start"])
        bars = load_bars(cfg.data_dir, symbol, start=LOAD_START[symbol], end=end)
        daily = resample_bars(bars, "1D").drop(columns="end")
        print(f"\n== {symbol}: daily {daily.index[0].date()} -> {daily.index[-1].date()} "
              f"({len(daily)} sessions), holdout sealed from {end.date()}")  # fmt: skip
        print("A. benchmark (10% vol target, carry + costs):")
        bench = {}
        for lb in (12, 3):
            for mult in (1.0, 3.0):
                r = benchmark(symbol, daily, lb, mult)
                bench[(lb, mult)] = r
                eras = "  ".join(
                    f"{n} {sharpe(r[(r.index >= a) & (r.index < b)]):+.2f}"
                    for n, a, b in ERAS[symbol]
                )
                print(f"  L={lb:<2} {mult:g}x  months={len(r)}  Sharpe {sharpe(r):+.2f}  "
                      f"ann.ret {r.mean() * 12:+.2%}  | {eras}")  # fmt: skip
                r.to_csv(out_dir / f"{symbol}_bench_L{lb}_{mult:g}x.csv")
        bh = buy_and_hold(symbol, daily, pd.DatetimeIndex(bench[(12, 1.0)].index))
        print(f"  buy-and-hold (excess, unscaled) Sharpe {sharpe(bh):+.2f}")

        r1, r3 = bench[(12, 1.0)], bench[(12, 3.0)]
        checks = {
            "1. Sharpe > 0.30": sharpe(r1) > 0.30,
            "2. positive every era": all(
                r1[(r1.index >= a) & (r1.index < b)].sum() > 0 for _, a, b in ERAS[symbol]
            ),
            "3. Sharpe > 0 at 3x": sharpe(r3) > 0,
        }
        if symbol == "USA500IDXUSD":
            checks["4. Sharpe >= B&H - 0.15"] = sharpe(r1) >= sharpe(bh) - 0.15
        if symbol == "EURUSD":
            print("B. bot run (PROPOSED risk, carry added per trade):")
            for lb in (12, 3):
                for equity in (25_000, 300_000):
                    for mult in (1.0, 2.0, 3.0):
                        m, t = bot_run(cfg, bars, lb, equity, mult, sha)
                        print(f"  L={lb:<2} ${equity:>7,} {mult:g}x trades={m['trades']:>3.0f} "
                              f"E[R]={m['expectancy_r']:+.3f} net={m['net_pnl']:>+9,.0f} "
                              f"(carry {m['carry_total']:+,.0f}) sharpe={m['sharpe']:+.2f} "
                              f"maxDD={m['max_drawdown']:.1%}")  # fmt: skip
                        if lb == 12 and equity == 25_000 and mult == 1.0:
                            checks["B. bot net > 0 and E[R] > 0 at $25k 1x"] = (
                                m["net_pnl"] > 0 and m["expectancy_r"] > 0
                            )
                            t.to_csv(out_dir / "EURUSD_bot_L12_25k_1x_trades.csv", index=False)
        verdicts[symbol] = checks
    print("\npre-registered verdict (L=12):")
    for symbol, checks in verdicts.items():
        v = "PASS" if all(checks.values()) else "FAIL"
        print(f"  {symbol}: {v}  " + "; ".join(f"{k}={'ok' if ok else 'NO'}"
                                               for k, ok in checks.items()))  # fmt: skip


if __name__ == "__main__":
    main()
