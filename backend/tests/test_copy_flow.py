"""End-to-end Phase-5 flow on in-memory SQLite: a source fill runs
detect→risk→paper-execute→portfolio; the kill switch halts execution; risk adjusts size."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.copy_trading.engine import CopyEngine
from app.copy_trading.executors import PaperExecutor
from app.database.models import CopyOrder, PortfolioPosition, Trade, Wallet
from app.portfolio.service import current_cash
from app.portfolio.valuation import revalue
from app.risk.engine import RiskEngine
from app.risk.killswitch import is_engaged, set_engaged
from app.risk.models import RiskLimits

WALLET = "0x" + "cc" * 20
NOW = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)


async def no_book(token_id: str) -> None:
    """Force the paper executor's book-less fallback for deterministic fills."""
    return None


def _engine(**kw) -> CopyEngine:
    return CopyEngine(executor=PaperExecutor(book_fetcher=no_book), **kw)


async def _add_wallet(session: AsyncSession, *, copy_enabled: bool = True) -> Wallet:
    wallet = Wallet(address=WALLET, is_tracked=True, copy_enabled=copy_enabled)
    session.add(wallet)
    await session.flush()
    return wallet


async def _add_trade(session: AsyncSession, side: str, price: str, size: str, **kw) -> Trade:
    trade = Trade(
        tx_hash="0x" + kw.pop("h", "01") * 32,
        log_index=0,
        wallet_address=WALLET,
        token_id="tok",
        side=side,
        price=Decimal(price),
        size=Decimal(size),
        usdc_amount=Decimal(price) * Decimal(size),
        timestamp=kw.pop("ts", NOW),
        block_number=1,
    )
    session.add(trade)
    await session.flush()
    return trade


async def test_buy_flows_detect_risk_execute_portfolio(session: AsyncSession) -> None:
    await _add_wallet(session)
    trade = await _add_trade(session, "BUY", "0.40", "100")

    order = await _engine().process_trade(session, trade, now=NOW)

    assert order.status == "filled"
    assert order.filled_size == Decimal("100.000000")
    assert order.fill_price == Decimal("0.402000")  # 0.40 * (1 + 0.005) fallback slippage
    assert order.slippage == Decimal("0.005000")
    assert order.source_trade_id == trade.id

    position = await session.scalar(
        select(PortfolioPosition).where(PortfolioPosition.token_id == "tok")
    )
    assert position is not None
    assert position.size == Decimal("100.000000")
    assert position.avg_entry_price == Decimal("0.402000")
    # 10_000 starting cash minus 100 * 0.402.
    assert await current_cash(session, "paper") == Decimal("9959.800000")


async def test_kill_switch_halts_execution(session: AsyncSession) -> None:
    await _add_wallet(session)
    trade = await _add_trade(session, "BUY", "0.40", "100")

    await set_engaged(session, True, "manual halt")
    assert await is_engaged(session) is True

    order = await _engine().process_trade(session, trade, now=NOW)
    assert order.status == "rejected"
    assert order.error == "kill switch engaged"
    # No position opened, cash untouched.
    assert await current_cash(session, "paper") == Decimal("10000.000000")


async def test_disabled_wallet_is_rejected(session: AsyncSession) -> None:
    await _add_wallet(session, copy_enabled=False)
    trade = await _add_trade(session, "BUY", "0.40", "100")

    order = await _engine().process_trade(session, trade, now=NOW)
    assert order.status == "rejected"
    assert "wallet copy disabled" in order.error


async def test_risk_adjusts_size_down(session: AsyncSession) -> None:
    await _add_wallet(session)
    trade = await _add_trade(session, "BUY", "0.50", "1000")  # proportional -> 1000 shares desired

    tight = RiskLimits.from_settings(get_settings())
    # No snapshot yet, so equity falls back to paper_starting_cash (10000); 0.5% of that is a
    # 50 notional cap -> 100 shares at 0.50.
    tight.max_position_size_pct = Decimal("0.005")
    order = await _engine(risk_engine=RiskEngine(limits=tight)).process_trade(
        session, trade, now=NOW
    )

    assert order.status == "filled"
    assert order.requested_size == Decimal("1000.000000")
    assert order.filled_size == Decimal("100.000000")  # 50 / 0.50


async def test_sell_mirrors_exit_and_realizes_pnl(session: AsyncSession) -> None:
    await _add_wallet(session)
    engine = _engine()
    buy = await _add_trade(session, "BUY", "0.40", "100", h="01")
    await engine.process_trade(session, buy, now=NOW)

    sell = await _add_trade(session, "SELL", "0.60", "100", h="02", ts=NOW + timedelta(hours=1))
    order = await engine.process_trade(session, sell, now=NOW + timedelta(hours=1))

    assert order.status == "filled"
    assert order.side == "SELL"
    position = await session.get(PortfolioPosition, 1)
    assert position is not None
    assert position.is_open is False
    # Sold 100 @ 0.60*(1-0.005)=0.597, entry 0.402 -> realized (0.597-0.402)*100 = 19.5.
    assert position.realized_pnl == Decimal("19.500000")


async def test_detect_and_copy_is_idempotent(session: AsyncSession) -> None:
    await _add_wallet(session)
    await _add_trade(session, "BUY", "0.40", "100")
    engine = _engine()

    first = await engine.detect_and_copy(session, now=NOW)
    second = await engine.detect_and_copy(session, now=NOW)
    assert len(first) == 1
    assert len(second) == 0  # already copied -> not reprocessed
    assert (await session.scalar(select(func.count()).select_from(CopyOrder))) == 1


async def test_detect_skips_stale_trades(session: AsyncSession) -> None:
    await _add_wallet(session)
    await _add_trade(
        session, "BUY", "0.40", "100", ts=NOW - timedelta(hours=1)
    )  # older than lookback
    orders = await _engine().detect_and_copy(session, now=NOW)
    assert orders == []


async def test_revalue_writes_snapshot(session: AsyncSession) -> None:
    await _add_wallet(session)
    trade = await _add_trade(session, "BUY", "0.40", "100")
    await _engine().process_trade(session, trade, now=NOW)

    snapshot = await revalue(session, "paper", NOW)
    await session.commit()
    assert snapshot.open_positions == 1
    assert snapshot.cash == Decimal("9959.800000")
    # Marked at the latest observed price 0.40 vs entry 0.402 -> small unrealized loss.
    assert snapshot.total_value == Decimal("9999.800000")
