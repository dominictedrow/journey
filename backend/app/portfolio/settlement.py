"""Module 10 — resolution settlement: close positions in resolved markets at true payout.

Without this, a position in a resolved market is marked at the last observed trade print
forever: winners never realize their $1, losers never fall to $0, and every consumer of the
snapshot series (loss-limit rules, percentage sizing, the equity curve) drifts from reality.

`settle_resolved_positions` refreshes just the markets we hold (targeted Gamma fetch — the
hourly sweep is active-only and drops markets the moment they close), then closes each open
position whose market is closed with outcomePrices at its payout price. The close is a
synthetic settlement `CopyOrder` + `apply_fill(SELL)`, so cash and realized PnL stay on the
single copy_orders spine every other module reconciles against.

Live-mode caveat: this realizes the outcome in *our books*. Actually redeeming the CTF
tokens on-chain (converting winning shares back to collateral) is not automated — do that
in the Polymarket UI for now.
"""

import logging
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.blockchain.normalize import load_token_market_map
from app.database.models import CopyOrder, Market, PortfolioPosition
from app.ingest.gamma import sync_markets_by_ids
from app.portfolio.service import apply_fill

logger = logging.getLogger(__name__)

SETTLEMENT_NOTE = "resolution settlement"


def _payout_price(market: Market, token_id: str | None) -> Decimal | None:
    """The resolved payout for `token_id`, or None when the market can't settle it yet."""
    if not market.closed or not market.outcome_prices or not market.token_ids:
        return None
    tokens = [str(t) for t in market.token_ids]
    try:
        index = tokens.index(str(token_id))
    except ValueError:
        return None
    try:
        payout = Decimal(str(market.outcome_prices[index]))
    except (IndexError, ArithmeticError, TypeError, ValueError):
        return None
    return min(max(payout, Decimal(0)), Decimal(1))


async def _open_positions_all_modes(session: AsyncSession) -> list[PortfolioPosition]:
    return list(
        (
            await session.scalars(
                select(PortfolioPosition).where(
                    PortfolioPosition.is_open.is_(True), PortfolioPosition.size > 0
                )
            )
        ).all()
    )


async def _recover_market_ids(session: AsyncSession, positions: list[PortfolioPosition]) -> None:
    """Fill null market_ids from the token→market map once markets have been synced.
    The map is only loaded when some position actually needs it."""
    if all(p.market_id is not None or p.token_id is None for p in positions):
        return
    token_map = await load_token_market_map(session)
    for position in positions:
        if position.market_id is None and position.token_id is not None:
            position.market_id = token_map.get(str(position.token_id))


async def settle_resolved_positions(
    session: AsyncSession, *, refresh: bool = True, now: datetime | None = None
) -> list[int]:
    """Settle every open position whose market has resolved; returns settled position ids.

    `refresh=False` skips the Gamma round-trip (tests, or a caller that just synced)."""
    now = now or datetime.now(UTC)
    positions = await _open_positions_all_modes(session)
    if not positions:
        return []

    market_ids = sorted({p.market_id for p in positions if p.market_id is not None})
    if refresh and market_ids:
        try:
            await sync_markets_by_ids(session, market_ids)
        except Exception:  # noqa: BLE001 — settlement retries next beat; stale is not fatal
            logger.exception("targeted market refresh failed; settling on stored state")

    await _recover_market_ids(session, positions)
    settled: list[int] = []
    for position in positions:
        market = (
            await session.get(Market, position.market_id)
            if position.market_id is not None
            else None
        )
        if market is None:
            continue
        payout = _payout_price(market, position.token_id)
        if payout is None:
            continue
        size = position.size
        session.add(
            CopyOrder(
                created_at=now,
                mode=position.mode,
                source_wallet=position.source_wallet,
                source_trade_id=None,
                market_id=position.market_id,
                token_id=position.token_id,
                side="SELL",
                requested_size=size,
                requested_price=payout,
                status="filled",
                filled_size=size,
                fill_price=payout,
                delay_ms=0,
                slippage=Decimal(0),
                error=SETTLEMENT_NOTE,
            )
        )
        await apply_fill(
            session,
            mode=position.mode,
            source_wallet=position.source_wallet,
            market_id=position.market_id,
            token_id=position.token_id or "",
            side="SELL",
            filled_size=size,
            fill_price=payout,
            now=now,
        )
        settled.append(position.id)
        logger.info(
            "settled position %d (%s) at payout %s x %s",
            position.id,
            position.token_id,
            payout,
            size,
        )
    await session.commit()
    return settled
