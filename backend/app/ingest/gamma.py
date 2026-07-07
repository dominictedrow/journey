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
        "active": bool(raw.get("active", True)),
        "closed": bool(raw.get("closed", False)),
        "end_date": _dt(raw.get("endDate")),
        "liquidity": _num(raw.get("liquidity")),
        "volume": _num(raw.get("volume")),
        "created_at": _dt(raw.get("createdAt")),
    }


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
            response = await client.get("/markets", params=params | {"offset": offset})
            response.raise_for_status()
            batch = response.json()
            rows = [r for r in (parse_market(m) for m in batch) if r is not None]
            # Distinct Gamma markets can share a conditionId page-to-page; dedup per batch.
            rows = list({r["id"]: r for r in rows}.values())
            if rows:
                stmt = pg_insert(Market).values(rows)
                update_cols = {c: getattr(stmt.excluded, c) for c in rows[0] if c != "id"}
                await session.execute(
                    stmt.on_conflict_do_update(index_elements=["id"], set_=update_cols)
                )
                await session.commit()
                count += len(rows)
            if len(batch) < PAGE_SIZE:
                return count
            offset += PAGE_SIZE
