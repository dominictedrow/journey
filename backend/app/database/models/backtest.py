from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BigIntPK, JsonB


class Backtest(Base):
    """A backtest run (Module 7). `metrics` holds cagr/sharpe/sortino/max_dd/profit_factor/
    win_rate/turnover; `equity_curve` and `trade_log` hold the replay output."""

    __tablename__ = "backtests"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    strategy_id: Mapped[int | None] = mapped_column(ForeignKey("strategies.id"), index=True)
    params: Mapped[dict] = mapped_column(JsonB)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metrics: Mapped[dict | None] = mapped_column(JsonB)
    equity_curve: Mapped[list | None] = mapped_column(JsonB)
    trade_log: Mapped[list | None] = mapped_column(JsonB)
