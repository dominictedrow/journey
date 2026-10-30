"""Performance metrics. Win rate is reported but never used for ranking (MASTER_PLAN v10)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a positive fraction."""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float(((peak - equity) / peak).max())


def compute_metrics(
    trades: pd.DataFrame, equity: pd.Series, initial_equity: float
) -> dict[str, float]:
    rets = equity.pct_change().dropna() if len(equity) > 1 else pd.Series(dtype=float)
    std = rets.std(ddof=1) if len(rets) > 1 else 0.0
    downside = rets[rets < 0]
    down_dev = math.sqrt((downside**2).sum() / len(rets)) if len(rets) else 0.0
    years = len(equity) / PERIODS_PER_YEAR if len(equity) else 0.0
    final = float(equity.iloc[-1]) if len(equity) else initial_equity
    start = float(equity.iloc[0]) if len(equity) else initial_equity

    pnl = trades["pnl"] if len(trades) else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    hold = (
        (trades["exit_time"] - trades["entry_time"]).dt.total_seconds().mean() / 3600
        if len(trades)
        else 0.0
    )
    total_return = final / initial_equity - 1
    return {
        "trades": float(len(trades)),
        "trades_per_year": len(trades) / years if years else 0.0,
        "net_pnl": float(pnl.sum()),
        "total_return": total_return,
        "cagr": (final / start) ** (1 / years) - 1 if years and final > 0 and start > 0 else 0.0,
        "expectancy": float(pnl.mean()) if len(pnl) else 0.0,
        "expectancy_r": float(trades["r_multiple"].mean()) if len(trades) else 0.0,
        "sharpe": float(rets.mean() / std * math.sqrt(PERIODS_PER_YEAR)) if std > 0 else 0.0,
        "sortino": (
            float(rets.mean() / down_dev * math.sqrt(PERIODS_PER_YEAR)) if down_dev > 0 else 0.0
        ),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": float(wins / losses) if losses > 0 else (np.inf if wins > 0 else 0.0),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "avg_hold_hours": float(hold),
        "total_cost": float(trades["cost"].sum()) if len(trades) else 0.0,
    }
