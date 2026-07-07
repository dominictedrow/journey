"""Module 7 — deterministic backtest engine.

`run_backtest` replays a set of source-wallet fills as copy trades against a simulated
portfolio: same config + same fills = same result, byte-stable (golden-file tested).
This core is reused by the Module-6 optimizer.

Price model: every input fill is a price observation for its token. The market price at
any instant is the latest observation at or before it (a left-continuous step function,
extended flat past the last observation). A copy placed `delay_seconds` after the source
fill executes at that stepped price; `slippage_pct` then moves the execution adversely
(BUY up, SELL down) and `fee_pct` is charged on the filled notional. Delay therefore
costs exactly what the observed price path says it costs.

Copy semantics: every source BUY opens/increases our copy position for that
(wallet, token), sized per `BacktestConfig.sizing_mode`. A source SELL is mirrored
proportionally — if the source closes 40% of what they hold, we close 40% of ours.
Fills that violate constraints (position limit, no cash, nothing to mirror) are logged
as skipped with the reason (Transparency principle).

Portfolio marking: open copy positions are marked at the token's latest observed price,
so the equity curve moves with the source cohort's own trade prints.
"""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.backtesting.metrics import compute_backtest_metrics
from app.backtesting.models import BacktestConfig, BacktestResult
from app.database.models import Backtest, Trade

QUANTUM = Decimal("0.000001")
PRICE_FLOOR = Decimal("0.001")  # outcome-share prices live in (0, 1)
PRICE_CEIL = Decimal("0.999")
MIN_SPEND = Decimal("0.01")  # below this a fill is dust; skip instead


def _dec(value: float | Decimal) -> Decimal:
    return Decimal(str(float(value))).quantize(QUANTUM)


def _f(value: float | Decimal) -> float:
    return float(_dec(value))


def _iso(ts: datetime) -> str:
    """Emit a UTC-aware ISO timestamp. SQLite (test suite) returns naive datetimes while a
    caller-supplied `config.start` is aware; normalizing here keeps the equity curve internally
    consistent so metrics never mix naive and aware. Already-aware inputs are unchanged."""
    return (ts if ts.tzinfo else ts.replace(tzinfo=UTC)).isoformat()


def _event_key(t: Trade) -> tuple:
    return (t.timestamp, t.block_number or 0, t.log_index, t.wallet_address)


@dataclass
class _Position:
    size: Decimal = Decimal(0)
    avg: Decimal = Decimal(0)


class _PricePath:
    """Left-continuous step function of a token's observed trade prices."""

    def __init__(self) -> None:
        self.times: list[datetime] = []
        self.prices: list[Decimal] = []

    def observe(self, ts: datetime, price: Decimal) -> None:
        self.times.append(ts)
        self.prices.append(price)

    def at(self, ts: datetime) -> Decimal:
        index = bisect_right(self.times, ts) - 1
        return self.prices[max(index, 0)]


def run_backtest(config: BacktestConfig, trades: Sequence[Trade]) -> BacktestResult:
    """Replay `trades` (the source wallets' fills, any order) under `config`."""
    ordered = sorted(trades, key=_event_key)

    paths: dict[str, _PricePath] = {}
    for t in ordered:
        paths.setdefault(t.token_id or "", _PricePath()).observe(t.timestamp, t.price)

    delay = timedelta(seconds=config.delay_seconds)
    weights = config.weights or {}
    cash = config.initial_capital
    positions: dict[tuple[str, str], _Position] = {}
    source_net: dict[tuple[str, str], Decimal] = {}
    last_price: dict[str, Decimal] = {}
    equity_curve: list[dict] = []
    trade_log: list[dict] = []

    def marked_equity() -> Decimal:
        held = sum(
            (p.size * last_price[token] for (_, token), p in positions.items() if p.size > 0),
            Decimal(0),
        )
        return cash + held

    if ordered:
        start_ts = config.start or ordered[0].timestamp
        equity_curve.append({"t": _iso(start_ts), "equity": _f(cash)})

    for t in ordered:
        token = t.token_id or ""
        key = (t.wallet_address, token)
        last_price[token] = t.price
        base_price = paths[token].at(t.timestamp + delay)
        entry: dict = {
            "t": _iso(t.timestamp),
            "wallet": t.wallet_address,
            "token": token,
            "market_id": t.market_id,
            "side": t.side,
            "source_price": _f(t.price),
            "source_size": _f(t.size),
            "status": "skipped",
            "reason": None,
            "exec_price": None,
            "size": None,
            "notional": None,
            "fee": None,
            "realized_pnl": None,
        }

        if t.side == "BUY":
            weight = weights.get(t.wallet_address, Decimal(1))
            if config.sizing_mode == "fixed":
                spend = config.sizing_value * weight
            elif config.sizing_mode == "percentage":
                spend = config.sizing_value * marked_equity() * weight
            else:  # proportional to the source trade's own notional
                source_notional = t.usdc_amount if t.usdc_amount is not None else t.price * t.size
                spend = source_notional * config.sizing_value * weight
            if config.max_position_size is not None:
                spend = min(spend, config.max_position_size)
            spend = min(spend, cash / (1 + config.fee_pct))

            position = positions.get(key)
            opens_new = position is None or position.size <= 0
            open_count = sum(1 for p in positions.values() if p.size > 0)
            if (
                opens_new
                and config.max_open_positions is not None
                and open_count >= config.max_open_positions
            ):
                entry["reason"] = "max_open_positions"
            elif spend < MIN_SPEND:
                entry["reason"] = "insufficient_cash"
            else:
                exec_price = min(
                    max(base_price * (1 + config.slippage_pct), PRICE_FLOOR), PRICE_CEIL
                )
                size = _dec(spend / exec_price)
                cost = _dec(size * exec_price)
                fee = _dec(cost * config.fee_pct)
                cash -= cost + fee
                position = positions.setdefault(key, _Position())
                position.avg = _dec(
                    (position.avg * position.size + exec_price * size) / (position.size + size)
                )
                position.size += size
                entry.update(
                    status="filled",
                    exec_price=_f(exec_price),
                    size=_f(size),
                    notional=_f(cost),
                    fee=_f(fee),
                )
            source_net[key] = source_net.get(key, Decimal(0)) + t.size
        else:  # SELL — mirror the fraction of the source position being closed
            src_held = source_net.get(key, Decimal(0))
            matched = min(t.size, src_held)
            source_net[key] = src_held - matched
            position = positions.get(key)
            if src_held <= 0:
                entry["reason"] = "no_source_position"
            elif position is None or position.size <= 0:
                entry["reason"] = "no_position"
            else:
                fraction = matched / src_held
                sell_size = position.size if fraction == 1 else _dec(position.size * fraction)
                exec_price = max(
                    min(base_price * (1 - config.slippage_pct), PRICE_CEIL), PRICE_FLOOR
                )
                proceeds = _dec(sell_size * exec_price)
                fee = _dec(proceeds * config.fee_pct)
                cash += proceeds - fee
                realized = _dec((exec_price - position.avg) * sell_size - fee)
                position.size -= sell_size
                entry.update(
                    status="filled",
                    exec_price=_f(exec_price),
                    size=_f(sell_size),
                    notional=_f(proceeds),
                    fee=_f(fee),
                    realized_pnl=_f(realized),
                )

        equity = marked_equity()
        entry["cash"] = _f(cash)
        entry["equity"] = _f(equity)
        trade_log.append(entry)
        equity_curve.append({"t": _iso(t.timestamp), "equity": _f(equity)})

    return BacktestResult(
        metrics=compute_backtest_metrics(equity_curve, trade_log, config.initial_capital),
        equity_curve=equity_curve,
        trade_log=trade_log,
    )


async def load_backtest_trades(session: AsyncSession, config: BacktestConfig) -> list[Trade]:
    """The source wallets' indexed fills inside the configured window."""
    query = select(Trade).where(Trade.wallet_address.in_(config.wallets))
    if config.start is not None:
        query = query.where(Trade.timestamp >= config.start)
    if config.end is not None:
        query = query.where(Trade.timestamp <= config.end)
    query = query.order_by(Trade.timestamp, Trade.block_number, Trade.log_index)
    return list((await session.scalars(query)).all())


async def execute_backtest(session: AsyncSession, backtest_id: int) -> Backtest:
    """Run a queued `backtests` row: load fills, replay, persist result/status."""
    row = await session.get(Backtest, backtest_id)
    if row is None:
        raise ValueError(f"backtest {backtest_id} not found")
    row.status = "running"
    row.started_at = datetime.now(UTC)
    await session.commit()
    try:
        config = BacktestConfig.model_validate(row.params)
        result = run_backtest(config, await load_backtest_trades(session, config))
        row.metrics = result.metrics
        row.equity_curve = result.equity_curve
        row.trade_log = result.trade_log
        row.status = "completed"
    except Exception as exc:  # noqa: BLE001 — status must reflect any failure
        row.status = "failed"
        row.metrics = {"error": str(exc)}
    row.finished_at = datetime.now(UTC)
    await session.commit()
    return row
