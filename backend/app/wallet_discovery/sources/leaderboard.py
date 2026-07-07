"""Leaderboard discovery: top traders by PnL and by volume, recent and lifetime windows."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingest.leaderboard import get_leaderboard
from app.wallet_discovery.sources.base import Candidate

# (orderBy, timePeriod) sweeps: recent + lifetime views of both profit and volume.
SWEEPS = [("PNL", "MONTH"), ("PNL", "ALL"), ("VOL", "MONTH"), ("VOL", "ALL")]


async def leaderboard_source(session: AsyncSession) -> list[Candidate]:
    candidates: list[Candidate] = []
    for order_by, time_period in SWEEPS:
        rows = await get_leaderboard(order_by=order_by, time_period=time_period)
        source = f"leaderboard:{order_by}:{time_period}"
        candidates += [
            Candidate(address=row["proxyWallet"].lower(), source=source)
            for row in rows
            if row.get("proxyWallet")
        ]
    return candidates
