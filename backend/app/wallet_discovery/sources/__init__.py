"""Discovery sources. Each is `async (session) -> list[Candidate]`; failures are isolated
per source by the discovery runner, so one flaky API never blocks the others."""

from app.wallet_discovery.sources.base import Candidate, Source
from app.wallet_discovery.sources.leaderboard import leaderboard_source
from app.wallet_discovery.sources.market_holders import market_holders_source

ALL_SOURCES: list[Source] = [leaderboard_source, market_holders_source]

__all__ = ["ALL_SOURCES", "Candidate", "Source", "leaderboard_source", "market_holders_source"]
