"""After-cost competition: every mode, same data, same costs, same risk rules, then a ranking.

Holdout: the most recent ``holdout_months`` are sealed on first run (reports/holdout.json) and
never loaded unless ``unseal_holdout=True``. The seal does not move when new data arrives.
Every run is appended to artifacts/experiments.jsonl (all attempted variants are logged).
"""

from __future__ import annotations

import json
import subprocess
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trader import __version__
from trader.backtest.engine import Backtester, BacktestResult
from trader.backtest.metrics import compute_metrics, max_drawdown
from trader.data.loaders import cache_path, load_bars, snapshot_metadata
from trader.instruments import get_instrument
from trader.modes import build_mode
from trader.risk import RiskConfig

RANK_KEYS = {"expectancy_r": False, "sharpe": False, "sortino": False, "max_drawdown": True}


@dataclass
class CompetitionConfig:
    data_dir: Path
    report_dir: Path
    artifact_dir: Path
    instruments: list[dict]
    modes: list[dict]
    cost_multipliers: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    initial_equity: float = 25_000.0
    holdout_months: int = 12
    periods: list[dict] = field(default_factory=list)
    risk: RiskConfig = field(default_factory=RiskConfig)

    @classmethod
    def from_toml(cls, path: Path) -> CompetitionConfig:
        raw = tomllib.loads(path.read_text())
        base = path.parent.parent  # configs/ lives in the repo root
        run = raw["run"]
        return cls(
            data_dir=(base / run["data_dir"]).resolve(),
            report_dir=(base / run.get("report_dir", "reports")).resolve(),
            artifact_dir=(base / run.get("artifact_dir", "artifacts")).resolve(),
            instruments=raw["instruments"],
            modes=raw["modes"],
            cost_multipliers=run.get("cost_multipliers", [1.0, 2.0, 3.0]),
            initial_equity=run.get("initial_equity", 25_000.0),
            holdout_months=run.get("holdout_months", 12),
            periods=raw.get("periods", []),
            risk=RiskConfig(**raw.get("risk", {})),
        )


def git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def holdout_start(cfg: CompetitionConfig, symbol: str) -> pd.Timestamp:
    """Return the sealed holdout boundary for ``symbol``, sealing it on first use."""
    seal = cfg.report_dir / "holdout.json"
    sealed = json.loads(seal.read_text()) if seal.exists() else {}
    if symbol not in sealed:
        last = pd.Timestamp(snapshot_metadata(cache_path(cfg.data_dir, symbol))["last"])
        boundary = (last - pd.DateOffset(months=cfg.holdout_months)).normalize()
        boundary = boundary.replace(day=1)  # month-aligned, rounded earlier
        sealed[symbol] = {"holdout_start": boundary.isoformat(), "sealed_at": _now()}
        seal.parent.mkdir(parents=True, exist_ok=True)
        seal.write_text(json.dumps(sealed, indent=2) + "\n")
    return pd.Timestamp(sealed[symbol]["holdout_start"])


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def buy_and_hold(bars: pd.DataFrame, initial_equity: float) -> dict[str, float]:
    """Passive 1x-notional reference (not traded through the engine, no stop, no costs)."""
    daily = bars["close"].resample("1D").last().dropna()
    equity = initial_equity * daily / daily.iloc[0]
    years = len(equity) / 252
    rets = equity.pct_change().dropna()
    return {
        "total_return": float(equity.iloc[-1] / initial_equity - 1),
        "cagr": float((equity.iloc[-1] / initial_equity) ** (1 / years) - 1) if years else 0.0,
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0,
        "max_drawdown": max_drawdown(equity),
    }


def yearly_stability(res: BacktestResult) -> tuple[float, int]:
    """Fraction of calendar years with positive P&L, and how many years were active."""
    eq = res.daily_equity
    if eq.empty:
        return 0.0, 0
    yearly = eq.groupby(eq.index.year).last().diff()
    yearly.iloc[0] = eq.groupby(eq.index.year).last().iloc[0] - res.initial_equity
    active = yearly[yearly != 0]
    return (float((active > 0).mean()) if len(active) else 0.0), int(len(active))


def rank(table: pd.DataFrame) -> pd.Series:
    """Composite rank (1 = best): mean of per-metric ranks. Win rate deliberately excluded."""
    ranks = [table[k].rank(ascending=asc, method="min") for k, asc in RANK_KEYS.items()]
    return pd.concat(ranks, axis=1).mean(axis=1).rank(method="min")


def run_competition(
    cfg: CompetitionConfig, unseal_holdout: bool = False, log=print
) -> tuple[pd.DataFrame, Path]:
    repo = Path(__file__).resolve().parents[2]
    sha = git_sha(repo)
    rows: list[dict] = []
    period_rows: list[dict] = []
    benchmarks: dict[str, dict] = {}
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    trades_dir = cfg.artifact_dir / f"competition_{stamp}"
    trades_dir.mkdir(parents=True, exist_ok=True)

    for spec in cfg.instruments:
        symbol = spec["symbol"]
        inst = get_instrument(symbol)
        boundary = holdout_start(cfg, symbol)
        end = spec.get("end") if unseal_holdout else boundary
        bars = load_bars(cfg.data_dir, symbol, start=spec.get("start"), end=end)
        meta = snapshot_metadata(cache_path(cfg.data_dir, symbol))
        seal_state = "UNSEALED" if unseal_holdout else "sealed"
        log(f"{symbol}: {len(bars):,} bars {bars.index[0]} -> {bars.index[-1]} "
            f"(holdout from {boundary.date()} {seal_state})")  # fmt: skip
        benchmarks[symbol] = buy_and_hold(bars, cfg.initial_equity)

        for mspec in cfg.modes:
            for mult in cfg.cost_multipliers:
                mode = build_mode(mspec["name"], **mspec.get("params", {}))
                if not mode.supports(symbol):
                    continue
                label = mspec.get("label", mode.name)
                bt = Backtester(inst, cfg.risk, cfg.initial_equity, cost_multiplier=mult)
                res = bt.run(mode, bars)
                m = compute_metrics(res.trades, res.daily_equity, cfg.initial_equity)
                pos_years, n_years = yearly_stability(res)
                row = {
                    "symbol": symbol, "mode": label, "provenance": mode.provenance.value,
                    "cost_x": mult, **m, "positive_years": pos_years, "active_years": n_years,
                    "rejections": res.rejections,
                }  # fmt: skip
                rows.append(row)
                if mult == 1.0:
                    period_rows += period_metrics(res, cfg.periods, symbol, label)
                res.trades.to_csv(trades_dir / f"{symbol}_{label}_{mult:g}x.csv", index=False)
                record = {
                    "time": _now(), "code": f"{__version__}@{sha}", "symbol": symbol,
                    "data_sha256": meta.get("source_sha256"), "start": str(bars.index[0]),
                    "end": str(bars.index[-1]), "holdout_unsealed": unseal_holdout,
                    "label": label, "mode": mode.describe(), "cost_x": mult, "risk": vars(cfg.risk),
                    "metrics": m,
                }  # fmt: skip
                with (cfg.artifact_dir / "experiments.jsonl").open("a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
                log(f"  {label:<18} {mult:g}x  trades={m['trades']:>6.0f} "
                    f"net={m['net_pnl']:>11,.0f}  sharpe={m['sharpe']:>5.2f} "
                    f"maxDD={m['max_drawdown']:.1%}")  # fmt: skip

    table = pd.DataFrame(rows)
    report = write_report(
        cfg, table, pd.DataFrame(period_rows), benchmarks, sha, stamp, unseal_holdout, trades_dir
    )
    return table, report


def period_metrics(res: BacktestResult, periods: list[dict], symbol: str, label: str) -> list[dict]:
    """Metrics restricted to each configured [start, end) window, from the same run."""
    out = []
    eq, trades = res.daily_equity, res.trades
    for p in periods:
        start = pd.Timestamp(p["start"]) if p.get("start") else eq.index.min()
        end = pd.Timestamp(p["end"]) if p.get("end") else eq.index.max() + pd.Timedelta("1D")
        window = eq[(eq.index >= start) & (eq.index < end)]
        if window.empty:
            continue
        before = eq[eq.index < start]
        base = float(before.iloc[-1]) if len(before) else res.initial_equity
        entry = trades["entry_time"].dt.tz_convert(None) if len(trades) else trades["entry_time"]
        t = trades[(entry >= start) & (entry < end)] if len(trades) else trades
        m = compute_metrics(t, pd.concat([pd.Series([base]), window]), base)
        out.append({"symbol": symbol, "mode": label, "period": p["name"], **m})
    return out


def _fmt_periods(df: pd.DataFrame) -> str:
    cols = ["period", "mode", "trades", "net_pnl", "expectancy_r", "sharpe", "max_drawdown"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples():
        lines.append(
            f"| {r.period} | {r.mode} | {r.trades:.0f} | {r.net_pnl:,.0f} | {r.expectancy_r:.3f} "
            f"| {r.sharpe:.2f} | {r.max_drawdown:.1%} |"
        )
    return "\n".join(lines) + "\n"


def _fmt_table(df: pd.DataFrame) -> str:
    cols = ["rank", "mode", "provenance", "trades", "trades_per_year", "net_pnl", "expectancy",
            "expectancy_r", "sharpe", "sortino", "max_drawdown", "profit_factor", "positive_years",
            "net_pnl_2x", "net_pnl_3x", "win_rate"]  # fmt: skip
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    lines = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if c in ("max_drawdown", "positive_years", "win_rate"):
                cells.append(f"{v:.1%}")
            elif c in ("net_pnl", "expectancy", "net_pnl_2x", "net_pnl_3x"):
                cells.append(f"{v:,.0f}" if c != "expectancy" else f"{v:,.2f}")
            elif isinstance(v, float):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def write_report(
    cfg: CompetitionConfig,
    table: pd.DataFrame,
    periods: pd.DataFrame,
    benchmarks: dict[str, dict],
    sha: str,
    stamp: str,
    unsealed: bool,
    trades_dir: Path,
) -> Path:
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.report_dir / f"competition_{stamp}.md"
    parts = [
        f"# Competition report {stamp}\n",
        f"- Code: `{__version__}@{sha}`; holdout {'**UNSEALED**' if unsealed else 'sealed'}",
        f"- Equity ${cfg.initial_equity:,.0f}; risk {vars(cfg.risk)} (PROPOSED, not approved)",
        f"- Cost multipliers {cfg.cost_multipliers}; costs are ASSUMPTIONS (`instruments.py`)",
        "- Rank = mean of ranks on expectancy (R), Sharpe, Sortino, max drawdown at 1x costs. "
        "Win rate is shown for context only.",
        f"- Trade ledgers: `{trades_dir}`\n",
    ]
    for symbol, grp in table.groupby("symbol", sort=False):
        base = grp[grp.cost_x == 1.0].set_index("mode").copy()
        for mult in (2.0, 3.0):
            alt = grp[grp.cost_x == mult].set_index("mode")["net_pnl"]
            base[f"net_pnl_{mult:g}x"] = alt.reindex(base.index).fillna(np.nan)
        base["rank"] = rank(base).astype(int)
        base = base.reset_index().sort_values("rank")
        bh = benchmarks[symbol]
        parts.append(f"## {symbol}\n")
        parts.append(
            f"Passive buy-and-hold reference (1x notional, no stop/costs): "
            f"return {bh['total_return']:.1%}, CAGR {bh['cagr']:.1%}, Sharpe {bh['sharpe']:.2f}, "
            f"max DD {bh['max_drawdown']:.1%}\n"
        )
        parts.append(_fmt_table(base))
        rej = {r["mode"]: r["rejections"] for _, r in grp[grp.cost_x == 1.0].iterrows()}
        parts.append("\nRisk-layer rejections at 1x: " + json.dumps(rej) + "\n")
        if not periods.empty:
            per = periods[(periods.symbol == symbol) & (periods.trades > 0)]
            if not per.empty:
                parts.append("### By period (1x costs)\n")
                parts.append(_fmt_periods(per))
    out.write_text("\n".join(parts))
    table.drop(columns=["rejections"]).to_csv(out.with_suffix(".csv"), index=False)
    return out
