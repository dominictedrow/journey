"""/wallets endpoints — wallet explorer, profile, tracking controls (Phase 2)."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from app.database.models import Position, Trade, Wallet, WalletMetrics
from app.deps import DbSession
from app.schemas.wallet import (
    TaskEnqueuedOut,
    WalletDetailOut,
    WalletListOut,
    WalletMetricsOut,
    WalletOut,
    WalletUpdate,
    WalletWithMetrics,
)
from app.wallet_analytics.metrics import compute_wallet_metrics

router = APIRouter(prefix="/wallets", tags=["wallets"])

RECENT_TRADES_LIMIT = 50

SORT_COLUMNS = {
    "total_volume": Wallet.total_volume,
    "total_trades": Wallet.total_trades,
    "last_active": Wallet.last_active,
    "discovered_at": Wallet.discovered_at,
    "roi_lifetime": WalletMetrics.roi_lifetime,
    "realized_pnl": WalletMetrics.realized_pnl,
    "win_rate": WalletMetrics.win_rate,
    "sharpe": WalletMetrics.sharpe,
}

SortKey = Literal[
    "total_volume",
    "total_trades",
    "last_active",
    "discovered_at",
    "roi_lifetime",
    "realized_pnl",
    "win_rate",
    "sharpe",
]


def _with_metrics(wallet: Wallet, metrics: WalletMetrics | None) -> WalletWithMetrics:
    out = WalletWithMetrics.model_validate(wallet)
    out.metrics = WalletMetricsOut.model_validate(metrics) if metrics else None
    return out


@router.get("", response_model=WalletListOut)
async def list_wallets(
    session: DbSession,
    tracked: bool | None = None,
    source: str | None = None,
    search: str | None = None,
    sort: SortKey = "total_volume",
    order: Literal["asc", "desc"] = "desc",
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> WalletListOut:
    query = select(Wallet, WalletMetrics).join(
        WalletMetrics, Wallet.address == WalletMetrics.wallet_address, isouter=True
    )
    if tracked is not None:
        query = query.where(Wallet.is_tracked == tracked)
    if source is not None:
        query = query.where(Wallet.discovery_source == source)
    if search:
        query = query.where(Wallet.address.contains(search.lower()))
    total = await session.scalar(select(func.count()).select_from(query.subquery())) or 0
    column = SORT_COLUMNS[sort]
    ordering = column.asc().nulls_last() if order == "asc" else column.desc().nulls_last()
    rows = await session.execute(
        query.order_by(ordering, Wallet.address).limit(limit).offset(offset)
    )
    items = [_with_metrics(wallet, metrics) for wallet, metrics in rows]
    return WalletListOut(items=items, total=total, limit=limit, offset=offset)


async def _get_wallet(session: DbSession, address: str) -> Wallet:
    wallet = await session.get(Wallet, address.lower())
    if wallet is None:
        raise HTTPException(status_code=404, detail="wallet not found")
    return wallet


@router.get("/{address}", response_model=WalletDetailOut)
async def get_wallet(session: DbSession, address: str) -> WalletDetailOut:
    wallet = await _get_wallet(session, address)
    metrics = await session.get(WalletMetrics, wallet.address)
    positions = (
        await session.scalars(
            select(Position)
            .where(Position.wallet_address == wallet.address, Position.is_open.is_(True))
            .order_by(Position.opened_at.desc())
        )
    ).all()
    trades = (
        await session.scalars(
            select(Trade)
            .where(Trade.wallet_address == wallet.address)
            .order_by(Trade.timestamp.desc(), Trade.log_index.desc())
            .limit(RECENT_TRADES_LIMIT)
        )
    ).all()
    base = _with_metrics(wallet, metrics)
    return WalletDetailOut(
        **base.model_dump(),
        open_positions=positions,  # type: ignore[arg-type]
        recent_trades=trades,  # type: ignore[arg-type]
    )


@router.patch("/{address}", response_model=WalletOut)
async def update_wallet(session: DbSession, address: str, update: WalletUpdate) -> Wallet:
    wallet = await _get_wallet(session, address)
    if update.is_tracked is not None:
        wallet.is_tracked = update.is_tracked
    if update.copy_enabled is not None:
        wallet.copy_enabled = update.copy_enabled
    await session.commit()
    return wallet


@router.post("/{address}/refresh-metrics", response_model=WalletWithMetrics)
async def refresh_metrics(session: DbSession, address: str) -> WalletWithMetrics:
    """Recompute metrics from already-indexed trades, inline (cheap: local DB only)."""
    wallet = await _get_wallet(session, address)
    metrics = await compute_wallet_metrics(session, wallet.address)
    return _with_metrics(wallet, metrics)


@router.post("/{address}/backfill", response_model=TaskEnqueuedOut, status_code=202)
async def backfill_wallet(session: DbSession, address: str) -> TaskEnqueuedOut:
    """Queue a full on-chain backfill + metrics compute for this wallet (Celery)."""
    from app.workers.celery_app import celery_app  # deferred: not needed at API import time

    wallet = await _get_wallet(session, address)
    result = celery_app.send_task("analytics.backfill_and_compute", args=[wallet.address])
    return TaskEnqueuedOut(task_id=result.id)
