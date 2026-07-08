"""Gamma API -> `markets` table sync."""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.models import Market
from app.ingest.http import throttled_get

PAGE_SIZE = 500


def _json_list(value: Any) -> list | None:
    """Gamma encodes list fields (outcomes, clobTokenIds) as JSON strings."""
    if isinstance(value, str):
        return json.loads(value) if value else None
    return value


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _num(value: Any) -> Decimal | None:
    return Decimal(str(value)) if value not in (None, "") else None


def parse_market(raw: dict) -> dict | None:
    """Map one Gamma market payload to a markets row; None if it has no condition_id."""
    if not raw.get("conditionId"):
        return None
    return {
        "id": raw["conditionId"],
        "question": raw.get("question") or "",
        "slug": raw.get("slug"),
        "category": raw.get("category"),
        "outcomes": _json_list(raw.get("outcomes")),
        "token_ids": _json_list(raw.get("clobTokenIds")),
        "outcome_prices": _json_list(raw.get("outcomePrices")),
        "active": bool(raw.get("active", True)),
        "closed": bool(raw.get("closed", False)),
        "neg_risk": bool(raw.get("negRisk", False)),
        "end_date": _dt(raw.get("endDate")),
        "liquidity": _num(raw.get("liquidity")),
        "volume": _num(raw.get("volume")),
        "created_at": _dt(raw.get("createdAt")),
    }


async def _upsert_markets(session: AsyncSession, rows: list[dict]) -> int:
    """Upsert parsed market rows (deduped by condition_id); commits. Returns rows written."""
    rows = list({r["id"]: r for r in rows}.values())
    if not rows:
        return 0
    stmt = pg_insert(Market).values(rows)
    update_cols = {c: getattr(stmt.excluded, c) for c in rows[0] if c != "id"}
    await session.execute(stmt.on_conflict_do_update(index_elements=["id"], set_=update_cols))
    await session.commit()
    return len(rows)


async def sync_markets(session: AsyncSession, active_only: bool = True) -> int:
    """Page through Gamma /markets and upsert; returns markets processed."""
    settings = get_settings()
    count = 0
    params: dict[str, Any] = {"limit": PAGE_SIZE}
    if active_only:
        params |= {"active": "true", "closed": "false"}
    async with httpx.AsyncClient(base_url=settings.gamma_api_url, timeout=30) as client:
        offset = 0
        while True:
            response = await throttled_get(
                client, "/markets", params | {"offset": offset}, upstream="gamma"
            )
            response.raise_for_status()
            batch = response.json()
            # Distinct Gamma markets can share a conditionId page-to-page; dedup per batch.
            count += await _upsert_markets(
                session, [r for r in (parse_market(m) for m in batch) if r is not None]
            )
            if len(batch) < PAGE_SIZE:
                return count
            offset += PAGE_SIZE


CONDITION_IDS_BATCH = 20  # condition_ids per targeted /markets request


async def sync_markets_by_ids(session: AsyncSession, condition_ids: list[str]) -> int:
    """Targeted refresh of specific markets (any state, including closed/resolved).

    The hourly `sync_markets` sweep is active-only, so a market we hold vanishes from it
    the moment it closes — exactly when settlement needs its outcomePrices. This pulls just
    those markets by condition_id; the caller keeps the list small (open positions), so the
    Gamma load is a request or two."""
    settings = get_settings()
    count = 0
    async with httpx.AsyncClient(base_url=settings.gamma_api_url, timeout=30) as client:
        for i in range(0, len(condition_ids), CONDITION_IDS_BATCH):
            batch_ids = condition_ids[i : i + CONDITION_IDS_BATCH]
            response = await throttled_get(
                client, "/markets", {"condition_ids": batch_ids}, upstream="gamma"
            )
            response.raise_for_status()
            batch = response.json()
            count += await _upsert_markets(
                session, [r for r in (parse_market(m) for m in batch) if r is not None]
            )
    return count
