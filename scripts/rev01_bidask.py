"""REV01 EUR/USD rerun on Dukascopy bid/ask data (D-020 / D-024 follow-up).

Same three pre-registered FX variants (07db1ec), same dev window (2004-01-01 -> sealed holdout),
same risk config. Only the data and cost treatment change:

  bid_0x        bid bars, zero cost        -> reproduces the original gross result
  mid_0x        mid bars, zero cost        -> gross edge on a quote-noise-free series
  mid_obs       mid bars, observed per-bar half-spread, no slippage   (Dukascopy ECN floor)
  mid_obs_slip  mid bars, observed half-spread + 0.2 pip slippage
  mid_assumed   mid bars, the original constant assumed costs (1.0 pip spread + 0.2 slippage)

Usage: uv run python scripts/rev01_bidask.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trader import __version__
from trader.backtest.engine import Backtester
from trader.backtest.metrics import compute_metrics
from trader.competition import CompetitionConfig, git_sha
from trader.data.loaders import BAR_COLUMNS, validate_bars
from trader.instruments import EURUSD
from trader.modes import build_mode

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / "data" / "dukascopy" / "EURUSD_1m_bidask.parquet"
START = pd.Timestamp("2004-01-01", tz="UTC")
VARIANTS = {
    "rev01_fx_5m": {"timeframe": "5min"},
    "rev01_fx_15m": {"timeframe": "15min"},
    "rev01_fx_15m_vwap": {"timeframe": "15min", "anchor": "vwap"},
}
SESSION = {"entry_start": "03:00", "entry_end": "12:00", "flat_time": "16:00"}
ERAS = [("2004-2011", "2004", "2011"), ("2012-2019", "2012", "2019"), ("2020 on", "2020", "2030")]


def bars_from(raw: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "bid":
        cols = {c: raw[f"bid_{c}"] for c in ("open", "high", "low", "close")}
    else:
        cols = {
            c: (raw[f"bid_{c}"] + raw[f"ask_{c}"]) / 2 for c in ("open", "high", "low", "close")
        }
    df = pd.DataFrame({**cols, "volume": raw["bid_volume"]})
    df.index = pd.DatetimeIndex(raw.index, name="ts")
    return validate_bars(df[BAR_COLUMNS].astype("float64"))


def gross_stats(trades: pd.DataFrame) -> tuple[float, float]:
    if trades.empty:
        return float("nan"), float("nan")
    bp = trades["pnl"] / (trades["units"] * trades["entry_price"]) * 1e4
    return float(bp.mean()), float(trades["r_multiple"].mean())


def main() -> None:
    cfg = CompetitionConfig.from_toml(REPO / "configs" / "competition.toml")
    sealed = json.loads((cfg.report_dir / "holdout.json").read_text())
    end = pd.Timestamp(sealed["EURUSD"]["holdout_start"])
    raw = pd.read_parquet(DATA)
    raw = raw[(raw.index >= START) & (raw.index < end)]
    print(f"EURUSD bid/ask {raw.index[0]} -> {raw.index[-1]} ({len(raw):,} bars), holdout sealed")
    bid, mid = bars_from(raw, "bid"), bars_from(raw, "mid")
    hs = ((raw["ask_open"] - raw["bid_open"]) / 2).clip(lower=0).to_numpy()
    print(f"observed half-spread at fills: median {np.median(hs) * 1e4:.2f} pip")

    no_slip = replace(EURUSD, slippage=0.0)
    scenarios = {
        "bid_0x": (EURUSD, bid, None, 0.0),
        "mid_0x": (EURUSD, mid, None, 0.0),
        "mid_obs": (no_slip, mid, hs, 1.0),
        "mid_obs_slip": (EURUSD, mid, hs, 1.0),
        "mid_assumed": (EURUSD, mid, None, 1.0),
    }
    sha = git_sha(REPO)
    rows, eras = [], []
    for label, params in VARIANTS.items():
        for scen, (inst, bars, spread, mult) in scenarios.items():
            mode = build_mode("rev01", **params, **SESSION)
            bt = Backtester(inst, cfg.risk, cfg.initial_equity, cost_multiplier=mult)
            res = bt.run(mode, bars, half_spread=spread)
            m = compute_metrics(res.trades, res.daily_equity, cfg.initial_equity)
            bp, r = gross_stats(res.trades)
            cost_pips = (
                (res.trades["cost"] / res.trades["units"]).mean() * 1e4 if len(res.trades) else 0
            )
            rows.append(
                {
                    "variant": label,
                    "scenario": scen,
                    "trades": int(m["trades"]),
                    "bp_per_trade": bp,
                    "r_per_trade": r,
                    "cost_pips_rt": cost_pips,
                    "return": m["net_pnl"] / cfg.initial_equity,
                    "max_dd": m["max_drawdown"],
                }
            )
            print(
                f"  {label:<18} {scen:<13} trades={m['trades']:>6.0f} bp/trade={bp:+.3f} "
                f"R={r:+.3f} cost={cost_pips:.2f}pip ret={rows[-1]['return']:+.1%}",
                flush=True,
            )
            if scen in ("bid_0x", "mid_0x", "mid_obs"):
                t = res.trades
                for name, a, b in ERAS:
                    sub = t[(t["entry_time"] >= a) & (t["entry_time"] < str(int(b) + 1))]
                    ebp, er = gross_stats(sub)
                    eras.append(
                        {"variant": label, "scenario": scen, "era": name, "trades": len(sub),
                         "bp_per_trade": ebp, "r_per_trade": er}
                    )  # fmt: skip
            rec = {
                "time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "code": f"{__version__}@{sha}",
                "symbol": "EURUSD",
                "data": "dukascopy bid/ask 1m (EURUSD_1m_bidask.parquet)",
                "start": str(bars.index[0]),
                "end": str(bars.index[-1]),
                "holdout_unsealed": False,
                "label": f"{label}:{scen}",
                "mode": mode.describe(),
                "cost_x": mult,
                "risk": vars(cfg.risk),
                "metrics": m,
            }
            with (cfg.artifact_dir / "experiments.jsonl").open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    out = cfg.artifact_dir / "rev01_bidask"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(eras).to_csv(out / "eras.csv", index=False)
    print("\nby era (gross / observed-spread):")
    print(pd.DataFrame(eras).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
