"""Module 10 unit tests: apply_fill averages up on repeat buys and realizes PnL on a
partial exit; cash is reconstructed from filled copy orders."""

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import CopyOrder
from app.portfolio.service import apply_fill, current_cash

MODE = "paper"
WALLET = "0x" + "ab" * 20


async def _fill(session: AsyncSession, side: str, size: str, price: str) -> None:
    """Record a filled copy order (for cash) and fold it into the portfolio."""
    session.add(
        CopyOrder(
            created_at=datetime.now(UTC),
            mode=MODE,
            source_wallet=WALLET,
            token_id="tok",
            side=side,
            requested_size=Decimal(size),
            filled_size=Decimal(size),
            fill_price=Decimal(price),
            status="filled",
        )
    )
    await apply_fill(
        session,
        mode=MODE,
        source_wallet=WALLET,
        market_id=None,
        token_id="tok",
        side=side,
        filled_size=Decimal(size),
        fill_price=Decimal(price),
    )
    await session.flush()


async def test_average_up_then_partial_exit(session: AsyncSession) -> None:
    await _fill(session, "BUY", "100", "0.40")
    await _fill(session, "BUY", "100", "0.60")
    position = await apply_fill(
        session,
        mode=MODE,
        source_wallet=WALLET,
        market_id=None,
        token_id="tok",
        side="BUY",
        filled_size=Decimal(0),
        fill_price=Decimal("0.50"),
    )
    assert position is not None
    assert position.size == Decimal("200.000000")
    assert position.avg_entry_price == Decimal("0.500000")  # (0.4*100 + 0.6*100) / 200

    # Sell half at 0.70 -> realized (0.70 - 0.50) * 100 = 20; position stays open at 100.
    await _fill(session, "SELL", "100", "0.70")
    assert position.is_open is True
    assert position.size == Decimal("100.000000")
    assert position.realized_pnl == Decimal("20.000000")


async def test_cash_tracks_buys_and_sells(session: AsyncSession) -> None:
    await _fill(session, "BUY", "100", "0.40")  # -40
    await _fill(session, "SELL", "50", "0.60")  # +30
    # 10_000 starting - 40 + 30.
    assert await current_cash(session, MODE) == Decimal("9990.000000")
