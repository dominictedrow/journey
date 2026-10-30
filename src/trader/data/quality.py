"""Data quality report for a cached 1-minute snapshot (Phase B2 gate artifact)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.data.resample import ET
from trader.sessions import nyse_sessions, rth_mask


def classify_gaps(index: pd.DatetimeIndex, min_gap: str = "30min") -> pd.DataFrame:
    """Gaps longer than ``min_gap``; ``weekend`` when the gap spans Friday->Sunday ET."""
    ts = index.to_series()
    gap = ts.diff()
    big = gap > pd.Timedelta(min_gap)
    g = pd.DataFrame({"last_bar": ts.shift(1)[big], "next_bar": ts[big], "gap": gap[big]})
    last_et = g["last_bar"].dt.tz_convert(ET)
    next_et = g["next_bar"].dt.tz_convert(ET)
    weekend = (
        (last_et.dt.dayofweek == 4) & (next_et.dt.dayofweek == 6) & (g["gap"] < pd.Timedelta("3D"))
    )
    daily_halt = g["gap"] <= pd.Timedelta("2h")
    g["kind"] = np.select([weekend, daily_halt], ["weekend", "daily_halt"], "other")
    return g.reset_index(drop=True)


def rth_coverage_by_year(index: pd.DatetimeIndex) -> pd.Series:
    """Share of NYSE regular-session minutes present, per year."""
    sessions = nyse_sessions(index[0], index[-1])
    expected = (
        ((sessions["close"] - sessions["open"]).dt.total_seconds() / 60)
        .groupby(sessions["open"].dt.year)
        .sum()
    )
    present = pd.Series(rth_mask(index), index=index.year).groupby(level=0).sum()
    return (present / expected).dropna()


def quality_report(bars: pd.DataFrame, symbol: str, meta: dict, equity_index: bool) -> str:
    idx = bars.index
    logret = np.log(bars["close"]).diff()
    gaps = classify_gaps(idx)
    flat = (bars["high"] == bars["low"]).mean()
    lines = [
        f"## {symbol}\n",
        f"- Source `{meta.get('source_file')}` sha256 `{meta.get('source_sha256', '')[:16]}…`",
        f"- Rows {len(bars):,}; {idx[0]} → {idx[-1]}; timezone {meta.get('timezone')}; "
        f"bar label {meta.get('bar_label')}",
        "- Duplicates 0, OHLC violations 0, non-positive prices 0 (enforced by loader)",
        f"- Flat bars (high == low) {flat:.1%}; zero-volume bars {(bars['volume'] == 0).sum():,}",
        f"- Gaps > 30 min: {len(gaps):,} total — "
        + ", ".join(f"{k} {v:,}" for k, v in gaps["kind"].value_counts().items()),
        "",
        "### Rows per year\n",
        "| year | rows | " + ("NYSE RTH coverage |" if equity_index else ""),
        "|---|---|" + ("---|" if equity_index else ""),
    ]
    per_year = pd.Series(1, index=idx.year).groupby(level=0).sum()
    cover = rth_coverage_by_year(idx) if equity_index else None
    for year, rows in per_year.items():
        cell = f" {cover.get(year, np.nan):.1%} |" if cover is not None else ""
        lines.append(f"| {year} | {rows:,} |{cell}")
    other = gaps[gaps["kind"] == "other"].sort_values("gap", ascending=False).head(12)
    lines += ["", "### Largest unexpected gaps\n", "| last bar | next bar | gap |", "|---|---|---|"]
    lines += [f"| {r.last_bar} | {r.next_bar} | {r.gap} |" for r in other.itertuples()]
    top = logret.abs().nlargest(10)
    lines += [
        "",
        "### Largest 1-minute moves\n",
        "| bar | log return | after gap? |",
        "|---|---|---|",
    ]
    gap_before = idx.to_series().diff()
    for ts in top.index:
        lines.append(f"| {ts} | {logret[ts]:+.4f} | {gap_before[ts] > pd.Timedelta('5min')} |")
    return "\n".join(lines) + "\n"
