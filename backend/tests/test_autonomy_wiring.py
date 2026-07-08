"""Autonomy wiring: strategy application, strategy-driven sizing, kill-switch auto-engage,
resolution settlement, autopromote selection, and the manual close endpoint."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config import get_settings
from app.copy_trading.engine import CopyEngine
from app.copy_trading.executors import PaperExecutor
from app.copy_trading.executors.base import ExecutionResult, Executor
from app.database.models import (
    CopyOrder,
    IndexerState,
    Market,
    PortfolioPosition,
    PortfolioSnapshot,
    Strategy,
    Trade,
    Wallet,
)
from app.portfolio.settlement import SETTLEMENT_NOTE, settle_resolved_positions
from app.risk.killswitch import is_engaged
from app.risk.models import TradeIntent
from app.strategy_optimizer.apply import apply_active_strategy
from app.wallet_discovery.autopromote import promotion_budget, promotion_candidates

NOW = datetime(2026, 7, 7, 12, 0, tzinfo=UTC)
W_A = "0x" + "aa" * 20
W_B = "0x" + "bb" * 20
W_C = "0x" + "cc" * 20


async def no_book(token_id: str) -> None:
    return None


def _wallet(address: str, **kw) -> Wallet:
    return Wallet(address=address, **kw)


def _strategy(params: dict, *, active: bool = True) -> Strategy:
    return Strategy(name="test strategy", params=params, is_active=active, created_at=NOW)


def _trade(wallet: str, side: str = "BUY", price: str = "0.50", size: str = "100") -> Trade:
    return Trade(
        tx_hash="0x" + "01" * 32,
        log_index=0,
        wallet_address=wallet,
        token_id="tok",
        side=side,
        price=Decimal(price),
        size=Decimal(size),
        usdc_amount=Decimal(price) * Decimal(size),
        timestamp=NOW,
        block_number=1,
    )


# --- apply_active_strategy -----------------------------------------------------------------


async def test_apply_enables_selected_and_demotes_others(session: AsyncSession) -> None:
    session.add_all(
        [
            _wallet(W_A, is_tracked=False, copy_enabled=False),
            _wallet(W_C, is_tracked=True, copy_enabled=True),
        ]
    )
    session.add(_strategy({"wallets": [W_A], "weights": {W_A: "1"}}))
    await session.commit()

    result = await apply_active_strategy(session)

    assert result["applied"] is True
    assert result["enabled"] == [W_A]
    assert result["demoted"] == [W_C]
    assert result["needs_backfill"] == [W_A]  # was untracked: seam must be re-derived
    a = await session.get(Wallet, W_A)
    c = await session.get(Wallet, W_C)
    assert a.copy_enabled is True and a.is_tracked is True and a.disabled_reason is None
    assert c.copy_enabled is False
    assert "rotated out by optimizer" in c.disabled_reason


async def test_apply_reenables_previously_rotated_out_wallet(session: AsyncSession) -> None:
    session.add(
        _wallet(
            W_A,
            is_tracked=True,
            copy_enabled=False,
            disabled_reason="rotated out by optimizer (strategy 1)",
        )
    )
    session.add(_strategy({"wallets": [W_A]}))
    await session.commit()

    result = await apply_active_strategy(session)

    assert result["enabled"] == [W_A]
    wallet = await session.get(Wallet, W_A)
    assert wallet.copy_enabled is True
    assert wallet.disabled_reason is None


async def test_apply_noop_without_active_or_resolved_strategy(session: AsyncSession) -> None:
    assert (await apply_active_strategy(session))["applied"] is False
    session.add(_strategy({"top_n": 5}))  # spec only — no resolved wallet set
    await session.commit()
    assert (await apply_active_strategy(session))["applied"] is False


# --- CopyEngine sizing from the active strategy ---------------------------------------------


async def test_copy_engine_sizes_from_active_strategy(session: AsyncSession) -> None:
    session.add(_wallet(W_A, is_tracked=True, copy_enabled=True))
    session.add(
        _strategy(
            {
                "sizing_mode": "fixed",
                "sizing_value": "50",
                "wallets": [W_A],
                "weights": {W_A: "2"},
            }
        )
    )
    trade = _trade(W_A, price="0.50", size="10")
    session.add(trade)
    await session.commit()

    engine = CopyEngine(executor=PaperExecutor(book_fetcher=no_book))
    order = await engine.process_trade(session, trade, now=NOW)

    # fixed 50 USDC * weight 2 = 100 USDC at 0.50 -> 200 shares requested.
    assert order.requested_size == Decimal("200.000000")


async def test_copy_engine_falls_back_to_settings_without_strategy(
    session: AsyncSession,
) -> None:
    session.add(_wallet(W_A, is_tracked=True, copy_enabled=True))
    trade = _trade(W_A, price="0.50", size="10")
    session.add(trade)
    await session.commit()

    engine = CopyEngine(executor=PaperExecutor(book_fetcher=no_book))
    order = await engine.process_trade(session, trade, now=NOW)

    # settings default: proportional 1.0x of the source's 5 USDC notional -> 10 shares.
    assert order.requested_size == Decimal("10.000000")


# --- kill-switch auto-engage on loss-limit breach --------------------------------------------


async def test_loss_limit_breach_engages_kill_switch(session: AsyncSession) -> None:
    session.add(_wallet(W_A, is_tracked=True, copy_enabled=True))
    # Equity dropped 200 within the day; the default daily-loss cap is 100.
    session.add_all(
        [
            PortfolioSnapshot(
                timestamp=NOW - timedelta(hours=2),
                mode="paper",
                cash=Decimal("10000"),
                total_value=Decimal("10000"),
            ),
            PortfolioSnapshot(
                timestamp=NOW - timedelta(minutes=1),
                mode="paper",
                cash=Decimal("9800"),
                total_value=Decimal("9800"),
            ),
        ]
    )
    trade = _trade(W_A)
    session.add(trade)
    await session.commit()

    engine = CopyEngine(executor=PaperExecutor(book_fetcher=no_book))
    order = await engine.process_trade(session, trade, now=NOW)

    assert order.status == "rejected"
    assert "loss" in (order.error or "")
    assert await is_engaged(session) is True  # halted durably, not just this order


# --- resolution settlement -------------------------------------------------------------------


async def test_settlement_closes_position_at_payout(session: AsyncSession) -> None:
    session.add(_wallet(W_A))
    session.add(
        Market(
            id="cond1",
            question="resolved?",
            token_ids=["tok", "tok-no"],
            outcome_prices=["1", "0"],
            active=False,
            closed=True,
        )
    )
    session.add(
        PortfolioPosition(
            mode="paper",
            market_id="cond1",
            token_id="tok",
            source_wallet=W_A,
            size=Decimal("100"),
            avg_entry_price=Decimal("0.40"),
            realized_pnl=Decimal(0),
            is_open=True,
            opened_at=NOW - timedelta(days=3),
        )
    )
    await session.commit()

    settled = await settle_resolved_positions(session, refresh=False, now=NOW)

    assert len(settled) == 1
    position = await session.scalar(select(PortfolioPosition))
    assert position.is_open is False
    assert position.size == Decimal("0")
    # (1.0 - 0.40) * 100 shares realized on the winning side.
    assert position.realized_pnl == Decimal("60.000000")
    order = await session.scalar(select(CopyOrder))
    assert order.error == SETTLEMENT_NOTE
    assert order.fill_price == Decimal("1")


async def test_settlement_skips_unresolved_and_recovers_market_id(
    session: AsyncSession,
) -> None:
    session.add(_wallet(W_A))
    session.add(
        Market(
            id="cond1",
            question="open still",
            token_ids=["tok"],
            outcome_prices=None,  # not resolved yet
            active=True,
            closed=False,
        )
    )
    session.add(
        PortfolioPosition(
            mode="paper",
            market_id=None,  # unknown at fill time; recovered from token_ids
            token_id="tok",
            source_wallet=W_A,
            size=Decimal("10"),
            avg_entry_price=Decimal("0.5"),
            is_open=True,
        )
    )
    await session.commit()

    settled = await settle_resolved_positions(session, refresh=False, now=NOW)

    assert settled == []
    position = await session.scalar(select(PortfolioPosition))
    assert position.market_id == "cond1"  # recovered, ready for the day it resolves
    assert position.is_open is True


# --- autopromote selection -------------------------------------------------------------------


async def test_promotion_candidates_skip_backfilled_and_order_by_discovery(
    session: AsyncSession,
) -> None:
    session.add_all(
        [
            _wallet(W_A, discovered_at=NOW - timedelta(hours=1)),
            _wallet(W_B, discovered_at=NOW - timedelta(days=2)),
            _wallet(W_C, discovered_at=NOW),  # already backfilled below
        ]
    )
    session.add(IndexerState(stream=f"history:{W_C}", last_block=123, updated_at=NOW))
    await session.commit()

    assert await promotion_candidates(session, 5) == [W_A, W_B]
    assert await promotion_candidates(session, 1) == [W_A]
    assert await promotion_candidates(session, 0) == []


async def test_promotion_budget_clamps_to_cohort_cap(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "autopromote_per_run", 8)
    monkeypatch.setattr(settings, "max_tracked_wallets", 2)
    session.add_all([_wallet(W_A, is_tracked=True), _wallet(W_B, is_tracked=False)])
    await session.commit()

    assert await promotion_budget(session) == 1  # cap 2 - 1 tracked


# --- manual close endpoint --------------------------------------------------------------------


class _StubExecutor(Executor):
    mode = "paper"

    async def execute(self, intent: TradeIntent, size: Decimal) -> ExecutionResult:
        return ExecutionResult("filled", size, intent.price, Decimal(0), 0)


@pytest.fixture
async def client(
    db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AsyncClient]:
    from app.api.routes import portfolio as portfolio_routes
    from app.deps import get_db_session
    from app.main import app

    monkeypatch.setattr(portfolio_routes, "PaperExecutor", _StubExecutor)
    maker = async_sessionmaker(db_engine, expire_on_commit=False)

    async def override() -> AsyncIterator:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_db_session] = override
    async with maker() as session:
        session.add(_wallet(W_A))
        session.add(
            PortfolioPosition(
                mode="paper",
                market_id=None,
                token_id="tok",
                source_wallet=W_A,
                size=Decimal("100"),
                avg_entry_price=Decimal("0.40"),
                realized_pnl=Decimal(0),
                is_open=True,
                opened_at=NOW,
            )
        )
        await session.commit()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
    app.dependency_overrides.clear()


async def test_close_position_endpoint(client: AsyncClient) -> None:
    response = await client.post("/portfolio/positions/1/close")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "filled"
    assert body["position"]["is_open"] is False
    # No trade prints for "tok", so the close executes at the avg-entry fallback price.
    assert Decimal(body["fill_price"]) == Decimal("0.40")

    # Second close is a conflict; unknown ids are 404.
    assert (await client.post("/portfolio/positions/1/close")).status_code == 409
    assert (await client.post("/portfolio/positions/999/close")).status_code == 404
