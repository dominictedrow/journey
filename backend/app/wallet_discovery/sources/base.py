from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class Candidate:
    """A wallet address proposed by a discovery source."""

    address: str  # lowercase 0x…
    source: str  # e.g. "leaderboard:PNL:MONTH", "market_holders"


Source = Callable[[AsyncSession], Awaitable[list[Candidate]]]
