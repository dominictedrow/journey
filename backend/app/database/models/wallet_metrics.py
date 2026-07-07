from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, DateTime, ForeignKey, Numeric
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, JsonB


class WalletMetrics(Base):
    """Latest computed metrics for a wallet (Module 3). One row per wallet."""

    __tablename__ = "wallet_metrics"

    wallet_address: Mapped[str] = mapped_column(ForeignKey("wallets.address"), primary_key=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Historical performance
    roi_lifetime: Mapped[Decimal | None] = mapped_column(Numeric)
    realized_pnl: Mapped[Decimal | None] = mapped_column(Numeric)
    unrealized_pnl: Mapped[Decimal | None] = mapped_column(Numeric)
    win_rate: Mapped[Decimal | None] = mapped_column(Numeric)
    avg_trade_return: Mapped[Decimal | None] = mapped_column(Numeric)
    profit_factor: Mapped[Decimal | None] = mapped_column(Numeric)
    cagr: Mapped[Decimal | None] = mapped_column(Numeric)

    # Recent performance & consistency
    roi_30d: Mapped[Decimal | None] = mapped_column(Numeric)
    roi_90d: Mapped[Decimal | None] = mapped_column(Numeric)
    monthly_consistency: Mapped[Decimal | None] = mapped_column(Numeric)
    weekly_consistency: Mapped[Decimal | None] = mapped_column(Numeric)

    # Drawdown / risk
    max_drawdown: Mapped[Decimal | None] = mapped_column(Numeric)
    losing_streak_max: Mapped[int | None] = mapped_column(BigInteger)
    volatility: Mapped[Decimal | None] = mapped_column(Numeric)
    concentration: Mapped[Decimal | None] = mapped_column(Numeric)
    sharpe: Mapped[Decimal | None] = mapped_column(Numeric)
    sortino: Mapped[Decimal | None] = mapped_column(Numeric)

    # Behavior
    avg_hold_hours: Mapped[Decimal | None] = mapped_column(Numeric)
    trade_frequency: Mapped[Decimal | None] = mapped_column(Numeric)
    avg_position_size: Mapped[Decimal | None] = mapped_column(Numeric)
    max_concurrent_positions: Mapped[int | None] = mapped_column(BigInteger)
    category_mix: Mapped[dict | None] = mapped_column(JsonB)

    # Edge signals
    early_entry_score: Mapped[Decimal | None] = mapped_column(Numeric)
    exit_timing_score: Mapped[Decimal | None] = mapped_column(Numeric)
    avg_prob_improvement: Mapped[Decimal | None] = mapped_column(Numeric)

    sample_size: Mapped[int | None] = mapped_column(BigInteger)
