"""/rankings endpoints — latest ranking generation with score breakdowns (Phase 3)."""

from typing import Literal

from fastapi import APIRouter, Query
from sqlalchemy import and_, func, select

from app.database.models import Ranking, Wallet, WalletScore
from app.deps import DbSession
from app.ranking.rankings import CATEGORIES, rebuild_rankings
from app.schemas.ranking import (
    CategoryCount,
    RankingEntryOut,
    RankingsOut,
    RebuildOut,
    WalletScoreOut,
)
from app.schemas.wallet import WalletOut

router = APIRouter(prefix="/rankings", tags=["rankings"])

RankingCategory = Literal[
    "overall",
    "politics",
    "crypto",
    "sports",
    "economics",
    "hft",
    "long_term",
    "early_info",
    "low_risk",
]


@router.get("", response_model=RankingsOut)
async def get_rankings(
    session: DbSession,
    category: RankingCategory = "overall",
    limit: int = Query(50, ge=1, le=200),
) -> RankingsOut:
    """Latest ranking generation for one category, each entry joined to its score row
    (same `computed_at`) so the breakdown/explanation ships with the rank."""
    latest = await session.scalar(select(func.max(Ranking.computed_at)))
    if latest is None:
        return RankingsOut(computed_at=None, category=category, categories=[], items=[])

    count_rows = await session.execute(
        select(Ranking.category, func.count())
        .where(Ranking.computed_at == latest)
        .group_by(Ranking.category)
    )
    counts: dict[str, int] = {category: count for category, count in count_rows.all()}
    categories = [CategoryCount(category=c, count=counts[c]) for c in CATEGORIES if c in counts]

    rows = await session.execute(
        select(Ranking, Wallet, WalletScore)
        .join(Wallet, Wallet.address == Ranking.wallet_address)
        .join(
            WalletScore,
            and_(
                WalletScore.wallet_address == Ranking.wallet_address,
                WalletScore.computed_at == Ranking.computed_at,
            ),
            isouter=True,
        )
        .where(Ranking.computed_at == latest, Ranking.category == category)
        .order_by(Ranking.rank)
        .limit(limit)
    )
    items = [
        RankingEntryOut(
            rank=ranking.rank,
            score=ranking.score,
            wallet=WalletOut.model_validate(wallet),
            score_detail=WalletScoreOut.model_validate(score) if score else None,
        )
        for ranking, wallet, score in rows
    ]
    return RankingsOut(computed_at=latest, category=category, categories=categories, items=items)


@router.post("/rebuild", response_model=RebuildOut)
async def rebuild(session: DbSession) -> RebuildOut:
    """Rescore the cohort and write a fresh generation, inline (cheap: local DB only)."""
    result = await rebuild_rankings(session)
    return RebuildOut(
        computed_at=result.computed_at,
        wallets_scored=result.wallets_scored,
        categories=result.category_counts,
    )
