"""Live-order reconciliation (Module 9, Phase 8).

A live copy can outlive the process that placed it: if the worker dies in the window between
the CLOB accepting an order and our transaction committing, the order exists on-chain with no
local record. `reconcile_live_orders` closes that gap. It enumerates the account's recent CLOB
orders and, for each:

  * matched to a tracked copy order still in flight — refreshes fill + status, folding a
    newly-observed fill into the portfolio exactly once;
  * with no local record (an *orphan*) — captures it as a `copy_orders` row tagged with the
    sentinel source wallet `"reconciled_orphan"` and auto-books any observed fill into the
    portfolio via `apply_fill`, so cash/positions stay accurate with zero human step (the whole
    point of the full-autonomy remodel: Dom never trades manually except kill-switch/close, so
    an orphan on this wallet is the bot's own order, not a stray human trade to preserve
    visibility into). A `risk_event` is still written for audit visibility either way.

The sweep is idempotent: captured orphans carry the CLOB order id, so a second run matches
them and skips (their status is terminal). Run on startup and on a slow beat.
"""

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.blockchain.normalize import load_token_market_map
from app.copy_trading.executors import OrderState, ReconcilableClient
from app.database.models import CopyOrder, RiskEvent
from app.portfolio.service import apply_fill

# Orders in these states are settled; the sweep leaves them alone.
TERMINAL = {"filled", "partial", "rejected", "skipped", "canceled", "orphaned"}

ORPHAN_SOURCE_WALLET = "reconciled_orphan"


async def reconcile_live_orders(
    session: AsyncSession, client: ReconcilableClient, *, now: datetime | None = None
) -> list[str]:
    """Reconcile live copy orders against the CLOB; returns the order ids touched. Commits."""
    now = now or datetime.now(UTC)
    remote = await client.list_recent_orders()

    known = {
        order.external_order_id: order
        for order in (
            await session.scalars(
                select(CopyOrder).where(
                    CopyOrder.mode == "live", CopyOrder.external_order_id.is_not(None)
                )
            )
        ).all()
    }

    touched: list[str] = []
    token_map: dict[str, str] | None = None
    for state in remote:
        if not state.order_id:
            continue
        order = known.get(state.order_id)
        if order is None:
            if token_map is None:
                token_map = await load_token_market_map(session)
            await _capture_orphan(session, state, now, token_map)
            touched.append(state.order_id)
            continue
        if order.status in TERMINAL:
            continue
        await _refresh_tracked(session, order, state, now)
        touched.append(state.order_id)

    await session.commit()
    return touched


async def _capture_orphan(
    session: AsyncSession, state: OrderState, now: datetime, token_map: dict[str, str]
) -> None:
    """Record a CLOB order with no local copy-order, auto-booking any observed fill."""
    market_id = token_map.get(str(state.token_id)) if state.token_id else None
    session.add(
        CopyOrder(
            created_at=now,
            mode="live",
            source_wallet=ORPHAN_SOURCE_WALLET,
            market_id=market_id,
            side=state.side or "",
            token_id=state.token_id,
            requested_size=state.filled_size or Decimal(0),
            filled_size=state.filled_size,
            fill_price=state.avg_price,
            status="orphaned",
            external_order_id=state.order_id,
            error="reconciled: live order on CLOB with no local record — auto-booked",
        )
    )

    bookable = (
        state.side is not None
        and state.token_id is not None
        and (state.filled_size or Decimal(0)) > 0
        and state.avg_price is not None
    )
    if bookable:
        await apply_fill(
            session,
            mode="live",
            source_wallet=ORPHAN_SOURCE_WALLET,
            market_id=market_id,
            token_id=state.token_id,  # type: ignore[arg-type]  # guarded by bookable
            side=state.side,  # type: ignore[arg-type]  # guarded by bookable
            filled_size=state.filled_size,
            fill_price=state.avg_price,  # type: ignore[arg-type]  # guarded by bookable
            now=now,
        )

    session.add(
        RiskEvent(
            timestamp=now,
            type="orphan_live_order",
            severity="warning",
            context={
                "order_id": state.order_id,
                "token_id": state.token_id,
                "side": state.side,
                "status": state.status,
                "filled_size": float(state.filled_size or 0),
                "avg_price": float(state.avg_price) if state.avg_price is not None else None,
                "auto_booked": bookable,
            },
            action_taken="auto_booked" if bookable else "flagged",
        )
    )


async def _refresh_tracked(
    session: AsyncSession, order: CopyOrder, state: OrderState, now: datetime
) -> None:
    """Update a still-in-flight tracked order, booking a newly-observed fill exactly once."""
    newly_filled = (
        state.status in ("filled", "partial")
        and (order.filled_size or Decimal(0)) <= 0
        and state.filled_size > 0
        and state.avg_price is not None
    )
    if newly_filled:
        await apply_fill(
            session,
            mode="live",
            source_wallet=order.source_wallet,
            market_id=order.market_id,
            token_id=order.token_id or "",
            side=order.side,
            filled_size=state.filled_size,
            fill_price=state.avg_price,  # type: ignore[arg-type]  # guarded by newly_filled
            now=now,
        )
        session.add(
            RiskEvent(
                timestamp=now,
                type="reconciled_fill",
                severity="info",
                context={"order_id": state.order_id, "filled_size": float(state.filled_size)},
                action_taken="booked",
            )
        )
    order.filled_size = state.filled_size
    order.fill_price = state.avg_price
    order.status = state.status
