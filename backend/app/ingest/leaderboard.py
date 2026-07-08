"""Trader leaderboard reads (discovery source).

GET {leaderboard_url}/v1/leaderboard — verified against
docs.polymarket.com/api-reference/core/get-trader-leaderboard-rankings on 2026-07-02.
Params: category, timePeriod (DAY/WEEK/MONTH/ALL), orderBy (PNL/VOL), limit (max 50), offset.
Rows carry `proxyWallet`, `userName`, `vol`, `pnl`, `rank`.
"""

import httpx

from app.config import get_settings
from app.ingest.http import throttled_get


async def get_leaderboard(
    order_by: str = "PNL",
    time_period: str = "MONTH",
    category: str = "OVERALL",
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    settings = get_settings()
    params: dict[str, str | int] = {
        "category": category,
        "timePeriod": time_period,
        "orderBy": order_by,
        "limit": limit or settings.discovery_leaderboard_limit,
        "offset": offset,
    }
    async with httpx.AsyncClient(base_url=settings.leaderboard_url, timeout=30) as client:
        response = await throttled_get(client, "/v1/leaderboard", params, upstream="data_api")
        response.raise_for_status()
        return response.json()
