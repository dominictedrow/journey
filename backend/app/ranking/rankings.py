"""Module 5 — ranking generations per category.

`rebuild_rankings` rescores the whole cohort (Module 4), assigns each scored wallet to
its categories, and writes one fresh `rankings` generation — every row of the generation
shares `computed_at` with the score rows it ranks, so the API can join score breakdowns
exactly. Old generations are retained as history; the API serves the latest one.

Category membership (docs/plan.md list):
- overall — every scored wallet.
- politics/crypto/sports/economics — the category dominates the wallet's traded volume
  (`category_mix` share >= CATEGORY_DOMINANCE).
- hft — trade frequency of at least HFT_MIN_TRADES_PER_DAY fills/day.
- long_term — average hold of at least LONG_TERM_MIN_HOLD_HOURS.
- early_info — early-entry score of at least EARLY_INFO_MIN_SCORE.
- low_risk — cohort-relative risk component of at least LOW_RISK_MIN_COMPONENT.

Within a category, wallets rank by composite score (ties broken by address for
determinism), capped at MAX_PER_CATEGORY rows.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adaptive_intelligence.scoring import score_all_wallets
from app.database.models import Ranking, WalletMetrics, WalletScore

CATEGORIES = (
    "overall",
    "politics",
    "crypto",
    "sports",
    "economics",
    "hft",
    "long_term",
    "early_info",
    "low_risk",
)
DOMAIN_CATEGORIES = ("politics", "crypto", "sports", "economics")

CATEGORY_DOMINANCE = 0.5  # share of traded volume in category_mix
HFT_MIN_TRADES_PER_DAY = Decimal("10")
LONG_TERM_MIN_HOLD_HOURS = Decimal("168")  # one week
EARLY_INFO_MIN_SCORE = Decimal("0.7")
LOW_RISK_MIN_COMPONENT = Decimal("0.7")
MAX_PER_CATEGORY = 100


@dataclass
class RebuildResult:
    computed_at: datetime
    wallets_scored: int
    category_counts: dict[str, int]


def categories_for(metrics: WalletMetrics | None, score: WalletScore) -> set[str]:
    """Every category this scored wallet belongs to."""
    cats = {"overall"}
    if metrics is None:
        return cats
    mix = metrics.category_mix or {}
    for category in DOMAIN_CATEGORIES:
        if mix.get(category, 0) >= CATEGORY_DOMINANCE:
            cats.add(category)
    if metrics.trade_frequency is not None and metrics.trade_frequency >= HFT_MIN_TRADES_PER_DAY:
        cats.add("hft")
    if metrics.avg_hold_hours is not None and metrics.avg_hold_hours >= LONG_TERM_MIN_HOLD_HOURS:
        cats.add("long_term")
    if metrics.early_entry_score is not None and metrics.early_entry_score >= EARLY_INFO_MIN_SCORE:
        cats.add("early_info")
    if score.risk_component is not None and score.risk_component >= LOW_RISK_MIN_COMPONENT:
        cats.add("low_risk")
    return cats


async def rebuild_rankings(session: AsyncSession) -> RebuildResult:
    """Rescore the cohort, then write a fresh rankings generation for every category."""
    now = datetime.now(UTC)
    score_rows = await score_all_wallets(session, now=now)
    metrics_by_address = {
        m.wallet_address: m for m in (await session.scalars(select(WalletMetrics))).all()
    }

    members: dict[str, list[WalletScore]] = {category: [] for category in CATEGORIES}
    for score in score_rows:
        for category in categories_for(metrics_by_address.get(score.wallet_address), score):
            members[category].append(score)

    category_counts: dict[str, int] = {}
    for category in CATEGORIES:
        ranked = sorted(members[category], key=lambda s: (-s.composite, s.wallet_address))
        ranked = ranked[:MAX_PER_CATEGORY]
        category_counts[category] = len(ranked)
        for rank, score in enumerate(ranked, start=1):
            session.add(
                Ranking(
                    computed_at=now,
                    category=category,
                    rank=rank,
                    wallet_address=score.wallet_address,
                    score=score.composite,
                )
            )
    await session.commit()
    return RebuildResult(
        computed_at=now, wallets_scored=len(score_rows), category_counts=category_counts
    )
