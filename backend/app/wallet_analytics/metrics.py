"""Module 3 — wallet analytics: deterministic metrics over a wallet's indexed fills.

`compute_metrics` is a pure function of (trades, markets, now): same inputs, same output —
golden-tested against fixtures. `compute_wallet_metrics` is the thin DB wrapper that loads
the inputs, upserts the `wallet_metrics` row, and refreshes the wallet's rollup columns.

Model notes (Phase-2 scope, documented limitations):
- PnL replays CLOB fills with the same average-cost model as `blockchain.normalize`.
  Redemptions of positions held to resolution are not fills, so that PnL is invisible
  until payout events are indexed (future work).
- Unrealized PnL marks each open position at the wallet's *own* last observed trade price
  for that token — a proxy until Phase-10 valuation supplies live order-book marks.
- A "realization" is one SELL fill matched against held size; win rate, profit factor,
  streaks, hold time, and the daily PnL series all derive from realization events.
"""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from math import sqrt

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Market, Trade, Wallet, WalletMetrics

QUANTUM = Decimal("0.000001")
DAYS_PER_YEAR = 365.25
MIN_CAGR_SPAN_DAYS = 30  # annualizing shorter histories produces absurd numbers


def _dec(value: float | Decimal | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(float(value))).quantize(QUANTUM)


def _mean(values: Sequence[Decimal]) -> Decimal | None:
    return _dec(sum(values, Decimal(0)) / len(values)) if values else None


@dataclass
class Realization:
    """One SELL fill matched against held size during the replay."""

    timestamp: datetime
    market_key: str  # market_id if known else token_id
    pnl: Decimal
    cost: Decimal  # avg_entry * matched size
    ret: Decimal | None  # pnl / cost
    hold_hours: float
    entry_price: Decimal
    exit_price: Decimal


@dataclass
class _TokenState:
    net: Decimal = Decimal(0)
    avg: Decimal = Decimal(0)
    opened_at: datetime | None = None


def _trade_notional(t: Trade) -> Decimal:
    return t.usdc_amount if t.usdc_amount is not None else t.price * t.size


def _replay(
    trades: Sequence[Trade], markets: dict[str, Market]
) -> tuple[list[Realization], dict[str, _TokenState], dict]:
    """Average-cost replay; returns realization events, final token state, and running stats."""
    state: dict[str, _TokenState] = {}
    events: list[Realization] = []
    stats: dict = {
        "buy_costs": [],  # notional per BUY fill
        "max_invested": Decimal(0),
        "max_concurrent": 0,
        "market_volume": {},  # market_key -> notional traded
        "category_volume": {},  # category -> notional traded
        "early_entry": [],  # per first market entry: 1 - elapsed/lifetime in [0, 1]
        "last_price": {},  # token -> last observed trade price
    }
    for t in sorted(trades, key=lambda t: (t.block_number or 0, t.log_index)):
        token = t.token_id or ""
        pos = state.setdefault(token, _TokenState())
        market = markets.get(t.market_id) if t.market_id else None
        market_key = t.market_id or token
        notional = _trade_notional(t)
        stats["last_price"][token] = t.price
        stats["market_volume"][market_key] = stats["market_volume"].get(market_key, 0) + notional
        category = (market.category if market else None) or "unknown"
        stats["category_volume"][category] = stats["category_volume"].get(category, 0) + notional
        if t.side == "BUY":
            stats["buy_costs"].append(notional)
            if pos.net == 0:
                pos.opened_at = t.timestamp
                if market and market.created_at and market.end_date:
                    lifetime = (market.end_date - market.created_at).total_seconds()
                    elapsed = (t.timestamp - market.created_at).total_seconds()
                    if lifetime > 0:
                        stats["early_entry"].append(
                            Decimal(str(min(max(1 - elapsed / lifetime, 0.0), 1.0)))
                        )
            pos.avg = ((pos.avg * pos.net + t.price * t.size) / (pos.net + t.size)).quantize(
                QUANTUM
            )
            pos.net += t.size
        else:
            matched = min(t.size, pos.net)
            if matched > 0:
                cost = pos.avg * matched
                pnl = (t.price - pos.avg) * matched
                hold = (
                    (t.timestamp - pos.opened_at).total_seconds() / 3600 if pos.opened_at else 0.0
                )
                events.append(
                    Realization(
                        timestamp=t.timestamp,
                        market_key=market_key,
                        pnl=pnl,
                        cost=cost,
                        ret=_dec(pnl / cost) if cost > 0 else None,
                        hold_hours=hold,
                        entry_price=pos.avg,
                        exit_price=t.price,
                    )
                )
                pos.net -= matched
        invested = sum((p.net * p.avg for p in state.values()), Decimal(0))
        stats["max_invested"] = max(stats["max_invested"], invested)
        stats["max_concurrent"] = max(
            stats["max_concurrent"], sum(1 for p in state.values() if p.net > 0)
        )
    return events, state, stats


def _daily_pnl_series(events: list[Realization], first_day: date, last_day: date) -> list[float]:
    """Realized PnL per calendar day over the active span (quiet days count as 0)."""
    by_day: dict[date, Decimal] = {}
    for e in events:
        by_day[e.timestamp.date()] = by_day.get(e.timestamp.date(), Decimal(0)) + e.pnl
    span = (last_day - first_day).days
    return [float(by_day.get(first_day + timedelta(days=i), 0)) for i in range(span + 1)]


def _consistency(events: list[Realization], key) -> Decimal | None:
    """Fraction of active periods (months/weeks) with positive realized PnL."""
    by_period: dict = {}
    for e in events:
        k = key(e.timestamp)
        by_period[k] = by_period.get(k, Decimal(0)) + e.pnl
    if not by_period:
        return None
    return _dec(sum(1 for v in by_period.values() if v > 0) / len(by_period))


def _window_roi(events: list[Realization], since: datetime) -> Decimal | None:
    window = [e for e in events if e.timestamp >= since]
    cost = sum((e.cost for e in window), Decimal(0))
    if cost <= 0:
        return None
    return _dec(sum((e.pnl for e in window), Decimal(0)) / cost)


def compute_metrics(trades: Sequence[Trade], markets: dict[str, Market], now: datetime) -> dict:
    """Every `wallet_metrics` column (except identity/computed_at) from a wallet's fills."""
    metrics: dict = {c: None for c in _METRIC_COLUMNS}
    metrics["sample_size"] = len(trades)
    if not trades:
        return metrics

    events, state, stats = _replay(trades, markets)
    ordered = sorted(trades, key=lambda t: (t.block_number or 0, t.log_index))
    first_ts, last_ts = ordered[0].timestamp, ordered[-1].timestamp
    span_days = max((last_ts - first_ts).total_seconds() / 86400, 1.0)

    realized = sum((e.pnl for e in events), Decimal(0))
    unrealized = sum(
        ((stats["last_price"][token] - pos.avg) * pos.net for token, pos in state.items()),
        Decimal(0),
    )
    total_buy_cost = sum(stats["buy_costs"], Decimal(0))

    metrics["realized_pnl"] = _dec(realized)
    metrics["unrealized_pnl"] = _dec(unrealized)
    if total_buy_cost > 0:
        roi = (realized + unrealized) / total_buy_cost
        metrics["roi_lifetime"] = _dec(roi)
        if span_days >= MIN_CAGR_SPAN_DAYS and roi > -1:
            metrics["cagr"] = _dec((1 + float(roi)) ** (DAYS_PER_YEAR / span_days) - 1)

    if events:
        wins = [e for e in events if e.pnl > 0]
        losses = [e for e in events if e.pnl < 0]
        metrics["win_rate"] = _dec(len(wins) / len(events))
        metrics["avg_trade_return"] = _mean([e.ret for e in events if e.ret is not None])
        gross_profit = sum((e.pnl for e in wins), Decimal(0))
        gross_loss = -sum((e.pnl for e in losses), Decimal(0))
        if gross_loss > 0:
            metrics["profit_factor"] = _dec(gross_profit / gross_loss)
        metrics["roi_30d"] = _window_roi(events, now - timedelta(days=30))
        metrics["roi_90d"] = _window_roi(events, now - timedelta(days=90))
        metrics["monthly_consistency"] = _consistency(events, lambda ts: (ts.year, ts.month))
        metrics["weekly_consistency"] = _consistency(events, lambda ts: ts.isocalendar()[:2])
        metrics["avg_hold_hours"] = _dec(statistics.fmean(e.hold_hours for e in events))
        metrics["avg_prob_improvement"] = _mean([e.exit_price - e.entry_price for e in events])

        streak = worst = 0
        for e in events:
            streak = streak + 1 if e.pnl < 0 else 0
            worst = max(worst, streak)
        metrics["losing_streak_max"] = worst

        equity = peak = trough_dd = Decimal(0)
        for e in events:
            equity += e.pnl
            peak = max(peak, equity)
            trough_dd = max(trough_dd, peak - equity)
        if stats["max_invested"] > 0:
            metrics["max_drawdown"] = _dec(trough_dd / stats["max_invested"])

        returns = [float(e.ret) for e in events if e.ret is not None]
        if len(returns) >= 2:
            metrics["volatility"] = _dec(statistics.pstdev(returns))
        daily = _daily_pnl_series(events, first_ts.date(), last_ts.date())
        if len(daily) >= 2:
            mean_d, std_d = statistics.fmean(daily), statistics.pstdev(daily)
            if std_d > 0:
                metrics["sharpe"] = _dec(mean_d / std_d * sqrt(365))
            downside = sqrt(statistics.fmean(min(x, 0.0) ** 2 for x in daily))
            if downside > 0:
                metrics["sortino"] = _dec(mean_d / downside * sqrt(365))

        # Exit timing: where each exit landed inside the token's observed price range.
        price_range: dict[str, tuple[Decimal, Decimal]] = {}
        for t in ordered:
            token = t.token_id or ""
            lo, hi = price_range.get(token, (t.price, t.price))
            price_range[token] = (min(lo, t.price), max(hi, t.price))
        timing = []
        for t in ordered:
            if t.side != "SELL":
                continue
            lo, hi = price_range[t.token_id or ""]
            if hi > lo:
                timing.append((t.price - lo) / (hi - lo))
        metrics["exit_timing_score"] = _mean(timing)

    metrics["trade_frequency"] = _dec(len(trades) / span_days)
    metrics["avg_position_size"] = _mean(stats["buy_costs"])
    metrics["max_concurrent_positions"] = stats["max_concurrent"]
    metrics["early_entry_score"] = _mean(stats["early_entry"])

    total_volume = sum(stats["market_volume"].values())
    if total_volume > 0:
        metrics["concentration"] = _dec(
            sum((v / total_volume) ** 2 for v in stats["market_volume"].values())
        )
        metrics["category_mix"] = {
            cat: float(_dec(v / total_volume) or 0)
            for cat, v in sorted(stats["category_volume"].items())
        }
    return metrics


_METRIC_COLUMNS = [
    c.name
    for c in WalletMetrics.__table__.columns
    if c.name not in ("wallet_address", "computed_at")
]


async def compute_wallet_metrics(session: AsyncSession, address: str) -> WalletMetrics:
    """Load a wallet's fills, compute metrics, upsert the row, refresh wallet rollups."""
    address = address.lower()
    trades = (
        await session.scalars(
            select(Trade).where(Trade.wallet_address == address).order_by(Trade.block_number)
        )
    ).all()
    market_ids = {t.market_id for t in trades if t.market_id}
    markets = (
        {
            m.id: m
            for m in (await session.scalars(select(Market).where(Market.id.in_(market_ids)))).all()
        }
        if market_ids
        else {}
    )

    now = datetime.now(UTC)
    row = await session.merge(
        WalletMetrics(
            wallet_address=address, computed_at=now, **compute_metrics(trades, markets, now)
        )
    )

    wallet = await session.get(Wallet, address)
    if wallet is not None and trades:
        wallet.total_trades = len(trades)
        wallet.total_volume = sum((_trade_notional(t) for t in trades), Decimal(0))
        wallet.first_seen = min(t.timestamp for t in trades)
        wallet.last_active = max(t.timestamp for t in trades)
    await session.commit()
    return row


async def wallets_with_trades(session: AsyncSession) -> list[str]:
    """Addresses that have at least one indexed fill (the metrics-refresh worklist)."""
    return list(
        (
            await session.scalars(
                select(Trade.wallet_address)
                .group_by(Trade.wallet_address)
                .order_by(func.min(Trade.id))
            )
        ).all()
    )
