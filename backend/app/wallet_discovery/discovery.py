"""Discovery runner: sweep every source and grow the candidate wallet pool.

New addresses are inserted with `discovery_source` + `discovered_at`; wallets we already
know (from any source, or auto-created by the chain indexer) are left untouched, so the
first attribution wins and re-runs are idempotent.
"""

import logging
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Wallet
from app.wallet_discovery.sources import ALL_SOURCES, Candidate, Source

logger = logging.getLogger(__name__)


async def upsert_candidates(
    session: AsyncSession, candidates: list[Candidate], discovered_at: datetime
) -> int:
    """Insert unseen candidate wallets; returns how many were actually new."""
    unique: dict[str, Candidate] = {}
    for c in candidates:
        unique.setdefault(c.address, c)  # first attribution wins within a batch too
    if not unique:
        return 0
    insert = pg_insert if session.bind.dialect.name == "postgresql" else sqlite_insert  # type: ignore[union-attr]
    rows = [
        {"address": c.address, "discovery_source": c.source, "discovered_at": discovered_at}
        for c in unique.values()
    ]
    result = await session.execute(insert(Wallet).values(rows).on_conflict_do_nothing())
    return cast(CursorResult[Any], result).rowcount or 0


async def run_discovery(
    session: AsyncSession, sources: list[Source] | None = None
) -> dict[str, int]:
    """Run each source independently; returns {source_fn_name: new_wallets}.

    A failing source is logged and skipped — the sweep never dies on one bad API.
    """
    now = datetime.now(UTC)
    results: dict[str, int] = {}
    for source in sources if sources is not None else ALL_SOURCES:
        name = source.__name__
        try:
            candidates = await source(session)
        except Exception:
            logger.exception("discovery source %s failed", name)
            continue
        results[name] = await upsert_candidates(session, candidates, now)
        await session.commit()
    return results
