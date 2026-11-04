"""``trader rebalance`` and ``trader rebalance-backtest``. Plans only; never sends orders."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from trader.data import yahoo
from trader.rebalance.backtest import simulate, summarize
from trader.rebalance.config import Holdings, RebalanceConfig
from trader.rebalance.planner import Plan, Quote, plan_rebalance

DATA_DIR = Path("../data/yahoo")


def latest_quotes(symbols: list[str], data_dir: Path) -> dict[str, Quote]:
    out = {}
    for s in symbols:
        path = data_dir / f"{s}.csv"
        if not path.exists():
            continue
        df = yahoo.load(s, data_dir)
        prev = float(df["close"].iloc[-2]) if len(df) > 1 else None
        out[s] = Quote(float(df["close"].iloc[-1]), df.index[-1].date(), prev)
    return out


def format_plan(plan: Plan, cfg: RebalanceConfig) -> str:
    lines = [f"Status: {plan.status}"]
    if plan.total_value:
        lines.append(f"Account value: ${plan.total_value:,.2f}")
        lines.append(f"{'':<6} {'now':>7} {'target':>7} {'drift':>7}")
        for s, w in plan.weights.items():
            tgt = cfg.rules.cash_target if s == "CASH" else cfg.targets.get(s, 0.0)
            flag = "  <- outside band" if abs(plan.drift[s]) > cfg.rules.band_abs else ""
            lines.append(f"{s:<6} {w:>7.1%} {tgt:>7.1%} {plan.drift[s]:>+7.1%}{flag}")
    if plan.trades:
        lines.append("\nProposed trades (NOT sent; place them yourself):")
        for t in plan.trades:
            lines.append(f"  {t.side:<4} {t.shares:>12.4f} {t.symbol:<5} @ ~${t.price:,.2f}"
                         f"  = ${t.value:,.2f}")  # fmt: skip
        lines.append(f"Cash after: ${plan.cash_after:,.2f}")
    lines += [f"- {m}" for m in plan.messages]
    return "\n".join(lines)


def run_plan(args) -> int:
    cfg = RebalanceConfig.from_toml(args.config)
    holdings = Holdings.from_toml(args.holdings)
    symbols = sorted(set(cfg.targets) | {s for s, n in holdings.shares.items() if n > 0})
    if args.refresh:
        failed = yahoo.download(symbols, args.data_dir, pause=0.5)
        if failed:
            print(f"Price refresh failed for {failed}; using any saved prices (checked for age).")
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    plan = plan_rebalance(cfg, holdings, latest_quotes(symbols, args.data_dir), today,
                          no_sell=args.no_sell)  # fmt: skip
    print(format_plan(plan, cfg))
    if plan.status == "HALT" and not args.refresh:
        print("Tip: rerun with --refresh to download current prices for every ticker you hold.")
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"plan_{today.isoformat()}.json"
    out.write_text(json.dumps({"date": today.isoformat(), **asdict(plan)}, indent=2, default=str))
    print(f"\nSaved: {out}")
    return 0 if plan.status != "HALT" else 2


def run_backtest(args) -> int:
    cfg = RebalanceConfig.from_toml(args.config)
    if cfg.rules.cash_target:
        print("note: cash earns nothing in this comparison")
    targets = dict(cfg.targets)
    px = pd.DataFrame({s: yahoo.load(s, args.data_dir)["adjclose"] for s in targets}).dropna()
    px = px[(px.index >= args.start) & (px.index < args.end)]
    print(f"{' + '.join(f'{w:.0%} {s}' for s, w in targets.items())}, "
          f"{px.index[0].date()} -> {px.index[-1].date()}, 5 bp per trade")  # fmt: skip
    print(f"{'rule':<14} {'CAGR':>6} {'vol':>6} {'Sharpe*':>8} {'maxDD':>6} {'rebal':>6} "
          f"{'turnover/yr':>11}")  # fmt: skip
    years = (px.index[-1] - px.index[0]).days / 365.25
    for rule, band in [("none", 0), ("annual", 0), ("monthly", 0), ("band", cfg.rules.band_abs)]:
        eq, n, turn = simulate(px, targets, rule, band or 0.05)
        s = summarize(eq)
        label = f"band ±{band:.0%}" if rule == "band" else rule
        print(f"{label:<14} {s['cagr']:>6.1%} {s['vol']:>6.1%} {s['sharpe_raw']:>8.2f} "
              f"{s['max_dd']:>6.1%} {n:>6} {turn / years:>11.1%}")  # fmt: skip
    print("*Sharpe on total returns (no cash rate subtracted): for comparing rules only.")
    return 0


def add_parsers(sub) -> None:
    p = sub.add_parser("rebalance", help="plan trades to bring holdings back to target (no orders)")
    p.add_argument("--config", type=Path, default=Path("configs/rebalance.toml"))
    p.add_argument("--holdings", type=Path, default=Path("configs/holdings.toml"))
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out", type=Path, default=Path("reports/rebalance"))
    p.add_argument("--refresh", action="store_true", help="download fresh prices first")
    p.add_argument("--no-sell", action="store_true", help="only use cash; never sell")
    p.add_argument("--today", help="override today's date (YYYY-MM-DD), for testing")
    p.set_defaults(func=run_plan)

    b = sub.add_parser("rebalance-backtest", help="compare rebalancing rules on history")
    b.add_argument("--config", type=Path, default=Path("configs/rebalance.toml"))
    b.add_argument("--data-dir", type=Path, default=DATA_DIR)
    b.add_argument("--start", default="2005-01-01")
    b.add_argument("--end", default="2025-10-01")
    b.set_defaults(func=run_backtest)
