"""Data-API reads: a wallet's positions/trades, used to seed and cross-check on-chain data."""

from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import get_settings

PAGE_SIZE = 1000  # limit accepted by /trades and /positions
# Polymarket caps offset-based pagination: offset > 3000 returns HTTP 400 regardless of limit.
# get_trades walks descending `end` time-windows to reach beyond this; _page_window stops
# gracefully at the cap so an unattended backfill never crashes on a high-volume wallet.
MAX_OFFSET = 3000


async def _page_window(client: httpx.AsyncClient, path: str, params: dict[str, Any]) -> list[dict]:
    """Offset-page one query up to the API's offset cap; returns what's reachable."""
    out: list[dict] = []
    offset = 0
    while True:
        response = await client.get(path, params=params | {"limit": PAGE_SIZE, "offset": offset})
        if response.status_code == 400 and offset > 0:
            break  # deep-pagination cap hit — return what we have (data is newest-first)
        response.raise_for_status()
        batch = response.json()
        out.extend(batch)
        if len(batch) < PAGE_SIZE or offset >= MAX_OFFSET:
            break
        offset += PAGE_SIZE
    return out


async def get_positions(address: str) -> list[dict]:
    settings = get_settings()
    async with httpx.AsyncClient(base_url=settings.data_api_url, timeout=30) as client:
        return await _page_window(client, "/positions", {"user": address.lower()})


async def iter_trades(address: str, end: int | None = None) -> AsyncIterator[list[dict]]:
    """Stream a wallet's trade history newest-first, one time-window (<= ~4000 fills) at a time.

    Offset paging caps at MAX_OFFSET, so we walk descending time windows via the `end` cursor
    (trades with timestamp <= end) to reach unlimited depth. Windows are non-overlapping
    (next end = oldest - 1), so no fill is yielded twice. Yielding per window keeps peak memory
    bounded to one window even for whale wallets with hundreds of thousands of fills, instead of
    materialising the whole history. Callers backfilling up to a handoff pass `end` so the API
    never even ships trades newer than the cutoff (the live tail already owns those).
    """
    settings = get_settings()
    async with httpx.AsyncClient(base_url=settings.data_api_url, timeout=30) as client:
        while True:
            params: dict[str, Any] = {"user": address.lower()}
            if end is not None:
                params["end"] = end
            window = await _page_window(client, "/trades", params)
            if not window:
                return
            yield window
            end = min(int(t["timestamp"]) for t in window) - 1  # strictly older next window


async def get_trades(address: str) -> list[dict]:
    """A wallet's full trade history, newest-first (fully materialised). Prefer `iter_trades`
    for unbounded backfills; this wrapper is for bounded callers and tests."""
    out: list[dict] = []
    async for window in iter_trades(address):
        out.extend(window)
    return out


async def get_holders(condition_ids: list[str], limit: int = 20) -> list[dict]:
    """Top holders per market token. GET /holders — verified against
    docs.polymarket.com/api-reference/core/get-top-holders-for-markets on 2026-07-02.

    Returns one object per token: {"token": ..., "holders": [{"proxyWallet": ...}, ...]}.
    `limit` is holders per token (API max 20).
    """
    settings = get_settings()
    params: dict[str, str | int] = {"market": ",".join(condition_ids), "limit": min(limit, 20)}
    async with httpx.AsyncClient(base_url=settings.data_api_url, timeout=30) as client:
        response = await client.get("/holders", params=params)
        response.raise_for_status()
        return response.json()
