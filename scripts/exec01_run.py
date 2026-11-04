"""EXEC01 run, exactly as pre-registered in research/EXEC01_DESIGN.md (b1dcdfe). One shot.

Usage: uv run python scripts/exec01_run.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trader import __version__
from trader.backtest.metrics import compute_metrics
from trader.backtest.passive import PassiveBacktester
from trader.competition import CompetitionConfig, git_sha
from trader.data.loaders import BAR_COLUMNS, validate_bars
from trader.instruments import EURUSD
from trader.modes import build_mode

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / "data" / "dukascopy" / "EURUSD_1m_bidask.parquet"
START = pd.Timestamp("2004-01-01", tz="UTC")
REV01_FX_5M = {"timeframe": "5min", "entry_start": "03:00", "entry_end": "12:00",
               "flat_time": "16:00"}  # fmt: skip
RETAIL_MIN_HALF = 0.00005  # 1.0 pip retail spread floor
ERAS = [("2004-2011", "2004", "2012"), ("2012-2019", "2012", "2020"), ("2020 on", "2020", "2030")]


def main() -> None:
    cfg = CompetitionConfig.from_toml(REPO / "configs" / "competition.toml")
    sealed = json.loads((cfg.report_dir / "holdout.json").read_text())
    end = pd.Timestamp(sealed["EURUSD"]["holdout_start"])
    raw = pd.read_parquet(DATA)
    raw = raw[(raw.index >= START) & (raw.index < end)]
    mid = pd.DataFrame(
        {c: (raw[f"bid_{c}"] + raw[f"ask_{c}"]) / 2 for c in ("open", "high", "low", "close")}
    )
    mid["volume"] = raw["bid_volume"]
    mid.index = pd.DatetimeIndex(raw.index, name="ts")
    mid = validate_bars(mid[BAR_COLUMNS].astype("float64"))
    obs = ((raw["ask_open"] - raw["bid_open"]) / 2).clip(lower=0).to_numpy()
    retail = np.maximum(obs, RETAIL_MIN_HALF)
    print(f"EURUSD mid {mid.index[0]} -> {mid.index[-1]} ({len(mid):,} bars), holdout sealed")

    scenarios = {  # name: (half-spread array, cost multiplier on slippage/sizing, equity, rule)
        "retail": (retail, 1.0, 25_000, "cross"),
        "ecn_floor": (obs, 0.0, 25_000, "cross"),
        "retail_2x": (retail * 2, 2.0, 25_000, "cross"),
        "retail_3x": (retail * 3, 3.0, 25_000, "cross"),
        "retail_300k": (retail, 1.0, 300_000, "cross"),
        "retail_through (sensitivity)": (retail, 1.0, 25_000, "through"),
    }
    sha = git_sha(REPO)
    out_dir = cfg.artifact_dir / "exec01"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, eras = [], []
    for variant, tp in (("E1", False), ("E2", True)):
        for scen, (hs, mult, equity, rule) in scenarios.items():
            inst = EURUSD.scaled_costs(mult) if mult else EURUSD.scaled_costs(0.0)
            bt = PassiveBacktester(inst, cfg.risk, equity, take_profit=tp, fill_rule=rule)
            res = bt.run(build_mode("rev01", **REV01_FX_5M), mid, hs)
            t, s = res.trades, res.signals
            m = compute_metrics(t, res.daily_equity, equity)
            fill_rate = float(s.filled.mean()) if len(s) else 0.0
            adv_f = float(s.loc[s.filled, "fwd_r"].mean())
            adv_u = float(s.loc[~s.filled, "fwd_r"].mean())
            row = {
                "variant": variant, "scenario": scen, "equity": equity,
                "signals": len(s), "fill_rate": fill_rate, "trades": int(m["trades"]),
                "trades_per_year": m["trades_per_year"], "expectancy_r": m["expectancy_r"],
                "total_return": m["total_return"], "max_dd": m["max_drawdown"],
                "fwd_r_filled": adv_f, "fwd_r_unfilled": adv_u,
                "exits": t.exit_reason.value_counts().to_dict() if len(t) else {},
            }  # fmt: skip
            rows.append(row)
            print(
                f"{variant} {scen:<29} signals={len(s):>6} fill={fill_rate:5.1%} "
                f"trades={m['trades']:>6.0f} E[R]={m['expectancy_r']:+.3f} "
                f"ret={m['total_return']:+8.1%} maxDD={m['max_drawdown']:5.1%} "
                f"fwdR filled/unfilled={adv_f:+.3f}/{adv_u:+.3f}",
                flush=True,
            )
            for name, a, b in ERAS:
                et = t[(t.entry_time >= a) & (t.entry_time < b)]
                es = s[(s.signal_time >= a) & (s.signal_time < b)]
                eras.append(
                    {"variant": variant, "scenario": scen, "era": name, "trades": len(et),
                     "expectancy_r": float(et.r_multiple.mean()) if len(et) else np.nan,
                     "pnl": float(et.pnl.sum()),
                     "fill_rate": float(es.filled.mean()) if len(es) else np.nan}
                )  # fmt: skip
            t.to_csv(out_dir / f"{variant}_{scen.split()[0]}_trades.csv", index=False)
            rec = {
                "time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "code": f"{__version__}@{sha}", "symbol": "EURUSD",
                "data": "dukascopy bid/ask 1m mid + scenario half-spread",
                "start": str(mid.index[0]), "end": str(mid.index[-1]),
                "holdout_unsealed": False, "label": f"exec01_{variant}:{scen}",
                "mode": {"rev01": REV01_FX_5M, "take_profit": tp, "fill_rule": rule},
                "cost_x": mult, "risk": vars(cfg.risk), "metrics": m,
                "fill_rate": fill_rate,
            }  # fmt: skip
            with (cfg.artifact_dir / "experiments.jsonl").open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    summary, era_df = pd.DataFrame(rows), pd.DataFrame(eras)
    summary.to_csv(out_dir / "summary.csv", index=False)
    era_df.to_csv(out_dir / "eras.csv", index=False)
    print("\nby era:")
    print(era_df.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print("\npre-registered pass criteria (retail, cross rule):")
    for v in ("E1", "E2"):
        r = summary[(summary.variant == v) & (summary.scenario == "retail")].iloc[0]
        r2 = summary[(summary.variant == v) & (summary.scenario == "retail_2x")].iloc[0]
        e = era_df[(era_df.variant == v) & (era_df.scenario == "retail")].set_index("era")
        checks = {
            "1. E[R] > 0.05 in 2012-19": e.loc["2012-2019", "expectancy_r"] > 0.05,
            "1. E[R] > 0.05 in 2020+": e.loc["2020 on", "expectancy_r"] > 0.05,
            "2. net > 0 at retail_2x": r2.total_return > 0,
            "3. fill rate >= 30%": r.fill_rate >= 0.30,
            "3. >= 300 trades/year": r.trades_per_year >= 300,
            "4. max DD < 25%": r.max_dd < 0.25,
        }
        verdict = "PASS" if all(checks.values()) else "FAIL"
        print(f"  {v}: {verdict}  " + "; ".join(f"{k}={'ok' if ok else 'NO'}"
                                              for k, ok in checks.items()))  # fmt: skip


if __name__ == "__main__":
    main()
