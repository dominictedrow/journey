"""Market-holders discovery: large position holders in the highest-volume active markets.

Surfaces conviction traders who never crack the global leaderboard (Data-API /holders
caps at 20 holders per token, so we sweep the top markets from our own `markets` sync).
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.models import Market
from app.ingest.data_api import get_holders
from app.wallet_discovery.sources.base import Candidate

HOLDERS_BATCH = 10  # condition_ids per /holders request


async def market_holders_source(session: AsyncSession) -> list[Candidate]:
    settings = get_settings()
    market_ids = (
        await session.scalars(
            select(Market.id)
            .where(Market.active.is_(True), Market.closed.is_(False))
            .order_by(Market.volume.desc().nulls_last())
            .limit(settings.discovery_holder_markets)
        )
    ).all()
    candidates: list[Candidate] = []
    for i in range(0, len(market_ids), HOLDERS_BATCH):
        batch = list(market_ids[i : i + HOLDERS_BATCH])
        for token in await get_holders(batch):
            candidates += [
                Candidate(address=h["proxyWallet"].lower(), source="market_holders")
                for h in token.get("holders", [])
                if h.get("proxyWallet")
            ]
    return candidates
