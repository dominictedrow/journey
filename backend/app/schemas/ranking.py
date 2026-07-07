"""Pydantic response models for /rankings (Decimal fields serialize as exact strings)."""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from app.schemas.wallet import WalletOut


class WalletScoreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    computed_at: datetime
    composite: Decimal
    historical_component: Decimal | None
    recent_component: Decimal | None
    copy_quality_component: Decimal | None
    risk_component: Decimal | None
    explanation: dict | None


class RankingEntryOut(BaseModel):
    rank: int
    score: Decimal
    wallet: WalletOut
    score_detail: WalletScoreOut | None


class CategoryCount(BaseModel):
    category: str
    count: int


class RankingsOut(BaseModel):
    computed_at: datetime | None  # None until the first rebuild
    category: str
    categories: list[CategoryCount]
    items: list[RankingEntryOut]


class RebuildOut(BaseModel):
    computed_at: datetime
    wallets_scored: int
    categories: dict[str, int]
