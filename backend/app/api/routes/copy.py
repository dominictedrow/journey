"""/copy endpoints — copy-order feed, run the detector, and simulate a source trade (Phase 5).

`/copy/simulate` is the manual trigger for the Phase-5 acceptance flow: inject one synthetic
source fill and watch it run detect→risk→paper-execute→portfolio, all synchronously.
"""

import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.copy_trading.engine import CopyEngine
from app.database.models import CopyOrder, Market, Trade, Wallet
from app.deps import DbSession
from app.portfolio.valuation import revalue
from app.schemas.copy import (
    CopyOrderListOut,
    CopyOrderOut,
    CopyStatusOut,
    SimulateResultOut,
    SimulateTradeIn,
    TaskEnqueuedOut,
)

router = APIRouter(prefix="/copy", tags=["copy"])


@router.get("", response_model=CopyStatusOut)
async def copy_status(session: DbSession, mode: str = "paper") -> CopyStatusOut:
    rows = await session.execute(
        select(CopyOrder.status, func.count())
        .where(CopyOrder.mode == mode)
        .group_by(CopyOrder.status)
    )
    by_status = {status: count for status, count in rows}
    last_order_at = await session.scalar(
        select(func.max(CopyOrder.created_at)).where(CopyOrder.mode == mode)
    )
    return CopyStatusOut(
        mode=mode,
        total_orders=sum(by_status.values()),
        by_status=by_status,
        last_order_at=last_order_at,
    )


@router.get("/orders", response_model=CopyOrderListOut)
async def list_orders(
    session: DbSession,
    mode: str | None = None,
    status: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> CopyOrderListOut:
    query = select(CopyOrder)
    if mode is not None:
        query = query.where(CopyOrder.mode == mode)
    if status is not None:
        query = query.where(CopyOrder.status == status)
    total = await session.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = (
        await session.scalars(query.order_by(CopyOrder.id.desc()).limit(limit).offset(offset))
    ).all()
    return CopyOrderListOut(
        items=[CopyOrderOut.model_validate(o) for o in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/run", response_model=TaskEnqueuedOut, status_code=202)
async def run_detector() -> TaskEnqueuedOut:
    """Queue a sweep of fresh source trades through the copy pipeline (Celery)."""
    from app.workers.celery_app import celery_app  # deferred: not needed at API import time

    result = celery_app.send_task("copy.detect_and_copy")
    return TaskEnqueuedOut(task_id=result.id)


@router.post("/simulate", response_model=SimulateResultOut, status_code=201)
async def simulate_trade(session: DbSession, body: SimulateTradeIn) -> SimulateResultOut:
    """Inject a synthetic source fill and run the full copy pipeline synchronously."""
    now = datetime.now(UTC)
    wallet_address = body.source_wallet.lower()

    wallet = await session.get(Wallet, wallet_address)
    if wallet is None:
        # A never-seen wallet is created copy-enabled so the happy path flows end to end.
        # Flush before inserting the trade that references it (FK: trades.wallet_address).
        session.add(
            Wallet(
                address=wallet_address,
                first_seen=now,
                last_active=now,
                discovery_source="simulate",
                discovered_at=now,
                is_tracked=True,
                copy_enabled=True,
            )
        )
        await session.flush()

    market_id = body.market_id
    if market_id is not None and await session.get(Market, market_id) is None:
        market_id = None  # avoid dangling FK to an unknown market

    trade = Trade(
        tx_hash=f"0x{secrets.token_hex(32)}",
        log_index=0,
        wallet_address=wallet_address,
        market_id=market_id,
        token_id=body.token_id,
        side=body.side,
        price=body.price,
        size=body.size,
        usdc_amount=body.usdc_amount if body.usdc_amount is not None else body.price * body.size,
        timestamp=now,
    )
    session.add(trade)
    await session.commit()

    order = await CopyEngine().process_trade(session, trade, now=now)
    await revalue(session, order.mode)
    await session.commit()

    detail = {
        "filled": f"paper-filled {order.filled_size} @ {order.fill_price}",
        "rejected": f"blocked by risk gate: {order.error}",
        "skipped": f"nothing to mirror: {order.error}",
        "partial": f"partial fill {order.filled_size} @ {order.fill_price}",
    }.get(order.status, order.status)
    return SimulateResultOut(order=CopyOrderOut.model_validate(order), detail=detail)
