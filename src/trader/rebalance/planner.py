"""Rebalance planner: holdings + prices -> a list of proposed trades. Plans only (D-031).

Never sends orders. Defaults to NO TRADE when any input is missing, stale or suspicious.

Algorithm:
1. Value every position at its latest close; account = positions + cash.
2. Drift = current weight - target weight, for each asset and for cash.
3. If no |drift| exceeds ``band_abs`` and nothing outside the allocation is held: no trades.
4. Otherwise trade every asset back to target. Sells are listed first, and buys are scaled
   down if sells + cash can't fund them. Trades under ``min_trade_usd`` are dropped. With
   whole shares only, quantities round toward zero (never overspend).
   ``no_sell=True`` uses cash only: buys go to the most underweight assets first.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

from trader.rebalance.config import Holdings, RebalanceConfig


@dataclass(frozen=True)
class Quote:
    price: float
    date: dt.date
    prev_close: float | None = None


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: str  # "BUY" or "SELL"
    shares: float
    price: float

    @property
    def value(self) -> float:
        return self.shares * self.price


@dataclass
class Plan:
    status: str  # "OK", "NO_TRADE_NEEDED" or "HALT"
    total_value: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    drift: dict[str, float] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    cash_after: float = 0.0
    messages: list[str] = field(default_factory=list)


def _halt(msg: str) -> Plan:
    return Plan(status="HALT", messages=[msg])


def check_quotes(cfg: RebalanceConfig, held: set[str], quotes: dict[str, Quote], today: dt.date):
    """Return a halt reason, or None if every needed quote is present, fresh and sane."""
    r = cfg.rules
    for sym in sorted(set(cfg.targets) | held):
        q = quotes.get(sym)
        if q is None or not math.isfinite(q.price) or q.price <= 0:
            return f"{sym}: no valid price"
        age = (today - q.date).days
        if age > r.max_price_age_days:
            return f"{sym}: price is {age} days old (limit {r.max_price_age_days})"
        if q.prev_close and abs(q.price / q.prev_close - 1) > r.max_daily_move:
            move = q.price / q.prev_close - 1
            return f"{sym}: one-day move {move:+.1%} exceeds {r.max_daily_move:.0%}; check data"
    return None


def plan_rebalance(
    cfg: RebalanceConfig,
    holdings: Holdings,
    quotes: dict[str, Quote],
    today: dt.date,
    no_sell: bool = False,
) -> Plan:
    r = cfg.rules
    held = {s for s, n in holdings.shares.items() if n > 0}
    reason = check_quotes(cfg, held, quotes, today)
    if reason:
        return _halt(f"NO TRADE: {reason}")

    symbols = sorted(set(cfg.targets) | held)
    values = {s: holdings.shares.get(s, 0.0) * quotes[s].price for s in symbols}
    total = sum(values.values()) + holdings.cash
    if total <= 0:
        return _halt("NO TRADE: account value is zero")

    weights = {s: values[s] / total for s in symbols}
    weights["CASH"] = holdings.cash / total
    targets = {s: cfg.targets.get(s, 0.0) for s in symbols} | {"CASH": r.cash_target}
    drift = {s: weights[s] - targets[s] for s in weights}
    plan = Plan(status="OK", total_value=total, weights=weights, drift=drift)

    if holdings.peak_value and total < holdings.peak_value * (1 - r.drawdown_alert):
        dd = 1 - total / holdings.peak_value
        plan.messages.append(f"ALERT: account is {dd:.1%} below its recorded peak")
    unknown = sorted(held - set(cfg.targets))
    if unknown:
        plan.messages.append(f"Not in allocation, will be sold to 0: {', '.join(unknown)}")

    if not unknown and max(abs(d) for d in drift.values()) <= r.band_abs:
        plan.status = "NO_TRADE_NEEDED"
        plan.cash_after = holdings.cash
        plan.messages.append(f"All weights within ±{r.band_abs:.0%} of target")
        return plan

    def qty(usd: float, price: float) -> float:
        q = usd / price
        return q if r.fractional else float(math.floor(q))

    deltas = {s: targets[s] * total - values[s] for s in symbols}
    cash = holdings.cash
    sells: list[Trade] = []
    if not no_sell:
        for s in symbols:
            if deltas[s] < 0:
                n = min(qty(-deltas[s], quotes[s].price), holdings.shares.get(s, 0.0))
                if s not in cfg.targets:
                    n = holdings.shares.get(s, 0.0)  # exit fully, even below min size
                t = Trade(s, "SELL", n, quotes[s].price)
                if n > 0 and (t.value >= r.min_trade_usd or s not in cfg.targets):
                    sells.append(t)
                    cash += t.value

    spendable = max(0.0, cash - r.cash_target * total)
    wants = {s: d for s, d in deltas.items() if d > 0}
    # no_sell: most underweight first, so limited cash goes where it's needed most.
    order = sorted(wants, key=lambda s: drift[s]) if no_sell else sorted(wants)
    need = sum(wants.values())
    scale = min(1.0, spendable / need) if need > 0 else 0.0
    buys: list[Trade] = []
    for s in order:
        usd = wants[s] * scale if not no_sell else min(wants[s], spendable)
        n = qty(usd, quotes[s].price)
        t = Trade(s, "BUY", n, quotes[s].price)
        if n > 0 and t.value >= r.min_trade_usd and t.value <= spendable + 1e-9:
            buys.append(t)
            spendable -= t.value
            cash -= t.value
    if scale < 1.0 and not no_sell:
        plan.messages.append(f"Buys scaled to {scale:.0%}: not enough cash after sells")

    plan.trades = sells + buys
    plan.cash_after = cash
    if not plan.trades:
        plan.status = "NO_TRADE_NEEDED"
        plan.messages.append("Drift exceeds the band but every trade is below min_trade_usd")
    return plan
