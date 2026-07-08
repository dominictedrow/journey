"""Phase 8 — gated live execution.

The `LiveExecutor` is exercised against a fake `LiveClient` (no py-clob-client, no network):
arming gates every order, a fill records execution quality, partials and retries behave, and
the copy engine mirrors a live buy→exit end to end. `reconcile_live_orders` recovers orphaned
and in-flight orders after a crash.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.copy_trading.engine import CopyEngine, select_executor
from app.copy_trading.executors import (
    LiveExecutor,
    OrderPlacement,
    OrderState,
    PaperExecutor,
)
from app.copy_trading.recovery import reconcile_live_orders
from app.database.models import CopyOrder, PortfolioPosition, RiskEvent, Trade, Wallet
from app.risk.engine import RiskEngine
from app.risk.killswitch import set_engaged
from app.risk.models import TradeIntent

WALLET = "0x" + "ab" * 20
NOW = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)


def armed_settings(**over) -> Settings:
    base = dict(
        execution_mode="live",
        live_trading_armed=True,
        live_retry_backoff_seconds=Decimal("0"),  # no real sleeps in tests
        # A funded live book: with no PortfolioSnapshot yet, total_equity falls back to this,
        # so the % exposure cap has room to allow trades (see test_live_buy_blocked_without_
        # starting_equity_or_snapshot for the zero-equity case this setting exists to avoid).
        risk_live_starting_equity=Decimal("10000"),
    )
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


class FakeLiveClient:
    """Records placement calls and returns queued `OrderPlacement`s (or raises)."""

    def __init__(self, responses=None, orders=None) -> None:
        self.calls: list[SimpleNamespace] = []
        self._responses = list(responses or [])
        self._orders = list(orders or [])

    async def place_order(self, *, token_id, side, price, size):
        self.calls.append(SimpleNamespace(token_id=token_id, side=side, price=price, size=size))
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    async def list_recent_orders(self):
        return list(self._orders)


def _intent(side="BUY", price="0.40", size="100") -> TradeIntent:
    return TradeIntent(
        source_wallet=WALLET,
        token_id="tok",
        side=side,  # type: ignore[arg-type]
        size=Decimal(size),
        price=Decimal(price),
        delay_seconds=2.0,
        mode="live",
    )


# --- Arming gate ----------------------------------------------------------------------------


async def test_disarmed_places_no_order() -> None:
    client = FakeLiveClient(responses=[OrderPlacement("o1", Decimal("100"), Decimal("0.40"))])
    executor = LiveExecutor(client=client, settings=armed_settings(live_trading_armed=False))

    result = await executor.execute(_intent(), Decimal("100"))

    assert result.status == "rejected"
    assert result.error == "live execution disarmed"
    assert client.calls == []  # nothing was ever sent to the venue


async def test_paper_mode_never_selects_live_executor() -> None:
    paper = Settings(execution_mode="paper", live_trading_armed=True)  # armed but not live mode
    assert isinstance(select_executor(paper), PaperExecutor)
    # Unarmed live also falls back to paper — live is strictly opt-in.
    assert isinstance(
        select_executor(Settings(execution_mode="live", live_trading_armed=False)), PaperExecutor
    )


# --- Fills + execution quality --------------------------------------------------------------


async def test_armed_buy_fills_and_records_quality() -> None:
    client = FakeLiveClient(responses=[OrderPlacement("o42", Decimal("100"), Decimal("0.402"))])
    executor = LiveExecutor(client=client, settings=armed_settings())

    result = await executor.execute(_intent("BUY", "0.40", "100"), Decimal("100"))

    assert result.status == "filled"
    assert result.filled_size == Decimal("100.000000")
    assert result.fill_price == Decimal("0.402000")
    assert result.slippage == Decimal("0.005000")  # (0.402-0.40)/0.40
    assert result.delay_ms == 2000
    assert result.external_order_id == "o42"
    # Priced marketable: a BUY crosses up by the 1% buffer -> 0.404.
    assert client.calls[0].price == Decimal("0.404")


async def test_partial_fill_reported() -> None:
    client = FakeLiveClient(responses=[OrderPlacement("o7", Decimal("40"), Decimal("0.41"))])
    executor = LiveExecutor(client=client, settings=armed_settings())

    result = await executor.execute(_intent("BUY", "0.40", "100"), Decimal("100"))

    assert result.status == "partial"
    assert result.filled_size == Decimal("40.000000")


async def test_unmatched_order_is_rejected() -> None:
    client = FakeLiveClient(responses=[OrderPlacement("o0", Decimal("0"), None)])
    executor = LiveExecutor(client=client, settings=armed_settings())

    result = await executor.execute(_intent(), Decimal("100"))
    assert result.status == "rejected"
    assert result.external_order_id == "o0"


async def test_transient_error_retries_then_fills() -> None:
    client = FakeLiveClient(
        responses=[
            RuntimeError("nonce too low"),
            OrderPlacement("o9", Decimal("100"), Decimal("0.40")),
        ]
    )
    executor = LiveExecutor(client=client, settings=armed_settings())

    result = await executor.execute(_intent("BUY", "0.40", "100"), Decimal("100"))
    assert result.status == "filled"
    assert len(client.calls) == 2  # retried once


async def test_gives_up_after_max_retries() -> None:
    client = FakeLiveClient(responses=[RuntimeError("boom")] * 3)
    executor = LiveExecutor(client=client, settings=armed_settings(live_max_retries=3))

    result = await executor.execute(_intent(), Decimal("100"))
    assert result.status == "rejected"
    assert "boom" in result.error
    assert len(client.calls) == 3


async def test_sell_marketable_price_crosses_down() -> None:
    client = FakeLiveClient(responses=[OrderPlacement("s1", Decimal("50"), Decimal("0.59"))])
    executor = LiveExecutor(client=client, settings=armed_settings())

    result = await executor.execute(_intent("SELL", "0.60", "50"), Decimal("50"))
    assert client.calls[0].price == Decimal("0.594")  # 0.60 * (1 - 0.01)
    assert result.slippage == Decimal("0.016667")  # -(0.59-0.60)/0.60, adverse positive


# --- End-to-end through the copy engine -----------------------------------------------------


async def _add_wallet(session: AsyncSession) -> None:
    session.add(Wallet(address=WALLET, is_tracked=True, copy_enabled=True))
    await session.flush()


async def _add_trade(session: AsyncSession, side, price, size, *, h="01", ts=NOW) -> Trade:
    trade = Trade(
        tx_hash="0x" + h * 32,
        log_index=0,
        wallet_address=WALLET,
        token_id="tok",
        side=side,
        price=Decimal(price),
        size=Decimal(size),
        usdc_amount=Decimal(price) * Decimal(size),
        timestamp=ts,
        block_number=1,
    )
    session.add(trade)
    await session.flush()
    return trade


async def test_live_engine_mirrors_buy_then_exit(session: AsyncSession) -> None:
    await _add_wallet(session)
    settings = armed_settings()
    client = FakeLiveClient(
        responses=[
            OrderPlacement("buy1", Decimal("100"), Decimal("0.402")),
            OrderPlacement("sell1", Decimal("100"), Decimal("0.597")),
        ]
    )
    engine = CopyEngine(executor=LiveExecutor(client=client, settings=settings), settings=settings)
    assert engine.mode == "live"

    buy = await _add_trade(session, "BUY", "0.40", "100", h="01")
    buy_order = await engine.process_trade(session, buy, now=NOW)
    assert buy_order.status == "filled"
    assert buy_order.mode == "live"
    assert buy_order.external_order_id == "buy1"

    sell = await _add_trade(session, "SELL", "0.60", "100", h="02", ts=NOW + timedelta(hours=1))
    sell_order = await engine.process_trade(session, sell, now=NOW + timedelta(hours=1))
    assert sell_order.status == "filled"
    assert sell_order.external_order_id == "sell1"

    position = await session.scalar(
        select(PortfolioPosition).where(
            PortfolioPosition.mode == "live", PortfolioPosition.token_id == "tok"
        )
    )
    assert position is not None
    assert position.is_open is False
    # Sold 100 @ 0.597, entry 0.402 -> realized (0.597-0.402)*100 = 19.5.
    assert position.realized_pnl == Decimal("19.500000")


async def test_live_buy_blocked_without_starting_equity_or_snapshot(session: AsyncSession) -> None:
    """Every exposure cap (position/market/wallet/total) is a % of equity; with no live
    snapshot yet and the starting equity fallback left at its zero default, every one of those
    caps is 0 and every live BUY is denied — the exact gap `risk_live_starting_equity` exists
    to let an operator configure their way out of once real capital backs the live wallet."""
    await _add_wallet(session)
    settings = armed_settings(risk_live_starting_equity=Decimal("0"))
    client = FakeLiveClient(responses=[OrderPlacement("buy1", Decimal("100"), Decimal("0.402"))])
    engine = CopyEngine(executor=LiveExecutor(client=client, settings=settings), settings=settings)

    buy = await _add_trade(session, "BUY", "0.40", "100", h="01")
    order = await engine.process_trade(session, buy, now=NOW)

    assert order.status == "rejected"
    # position_size is the tightest (and earliest-evaluated) of the equity-scaled caps, so it
    # denies first — any of them denying proves the zero-equity gap blocks every live BUY.
    assert order.error == "max_position_size_pct"
    assert client.calls == []  # risk gate stops us before the venue


async def test_kill_switch_blocks_live_order(session: AsyncSession) -> None:
    await _add_wallet(session)
    settings = armed_settings()
    client = FakeLiveClient(responses=[OrderPlacement("nope", Decimal("100"), Decimal("0.40"))])
    engine = CopyEngine(
        executor=LiveExecutor(client=client, settings=settings),
        risk_engine=RiskEngine(),
        settings=settings,
    )
    await set_engaged(session, True, "halt")

    trade = await _add_trade(session, "BUY", "0.40", "100")
    order = await engine.process_trade(session, trade, now=NOW)

    assert order.status == "rejected"
    assert order.error == "kill switch engaged"
    assert client.calls == []  # risk gate stops us before the venue


# --- Recovery / reconciliation --------------------------------------------------------------


async def test_reconcile_captures_orphan_and_is_idempotent(session: AsyncSession) -> None:
    client = FakeLiveClient(
        orders=[OrderState("orph1", "tok", "BUY", "filled", Decimal("50"), Decimal("0.5"))]
    )

    touched = await reconcile_live_orders(session, client, now=NOW)
    assert touched == ["orph1"]

    order = await session.scalar(select(CopyOrder).where(CopyOrder.external_order_id == "orph1"))
    assert order is not None
    assert order.status == "orphaned"
    event = await session.scalar(select(RiskEvent).where(RiskEvent.type == "orphan_live_order"))
    assert event is not None
    assert event.action_taken == "auto_booked"
    # A fillable orphan is auto-booked into the portfolio under the sentinel wallet.
    position = await session.scalar(select(PortfolioPosition))
    assert position is not None
    assert position.source_wallet == "reconciled_orphan"
    assert position.size == Decimal("50.000000")

    # A second sweep matches the captured orphan and does nothing new.
    again = await reconcile_live_orders(session, client, now=NOW)
    assert again == []
    count = len((await session.scalars(select(CopyOrder))).all())
    assert count == 1


async def test_reconcile_books_tracked_pending_fill(session: AsyncSession) -> None:
    await _add_wallet(session)
    session.add(
        CopyOrder(
            created_at=NOW,
            mode="live",
            source_wallet=WALLET,
            token_id="tok",
            side="BUY",
            requested_size=Decimal("100"),
            status="pending",
            external_order_id="track1",
        )
    )
    await session.commit()

    client = FakeLiveClient(
        orders=[OrderState("track1", "tok", "BUY", "filled", Decimal("100"), Decimal("0.40"))]
    )
    touched = await reconcile_live_orders(session, client, now=NOW)
    assert touched == ["track1"]

    order = await session.scalar(select(CopyOrder).where(CopyOrder.external_order_id == "track1"))
    assert order.status == "filled"
    assert order.filled_size == Decimal("100")
    position = await session.scalar(
        select(PortfolioPosition).where(PortfolioPosition.mode == "live")
    )
    assert position is not None
    assert position.size == Decimal("100.000000")
