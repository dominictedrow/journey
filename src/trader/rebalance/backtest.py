"""Compare rebalancing rules on history (reference only, not a strategy test).

Daily simulation on adjusted closes (dividends reinvested), cash earns nothing (conservative),
5 bp one-way cost on every rebalance trade. Rules:
  none     buy once, never rebalance
  annual   back to target on the first trading day of each year
  monthly  back to target on the first trading day of each month
  band     back to target on any day a weight drifts more than ``band`` from target
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

COST = 0.0005


def simulate(prices: pd.DataFrame, targets: dict[str, float], rule: str, band: float = 0.05):
    px = prices[list(targets)].dropna()
    w_t = np.array([targets[s] for s in px.columns])
    rets = px.pct_change().fillna(0.0).to_numpy()
    dates = px.index
    value = 1.0
    hold = w_t * value * (1 - COST)  # initial purchase
    values, n_reb, turnover = [], 0, 0.0
    for i in range(len(dates)):
        if i > 0:
            hold = hold * (1 + rets[i])
        value = hold.sum()
        w = hold / value
        new_period = i > 0 and (
            (rule == "annual" and dates[i].year != dates[i - 1].year)
            or (rule == "monthly" and dates[i].month != dates[i - 1].month)
        )
        if new_period or (rule == "band" and np.abs(w - w_t).max() > band):
            traded = np.abs(w_t * value - hold).sum()
            value -= traded * COST
            hold = w_t * value
            n_reb += 1
            turnover += traded / value
        values.append(value)
    return pd.Series(values, index=dates), n_reb, turnover


def summarize(eq: pd.Series) -> dict[str, float]:
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return {
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
        "vol": float(r.std() * math.sqrt(252)),
        "sharpe_raw": float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0,
        "max_dd": float((1 - eq / eq.cummax()).max()),
    }
