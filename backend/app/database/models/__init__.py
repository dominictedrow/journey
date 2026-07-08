"""All ORM models. Importing this package registers every table on Base.metadata
(required for Alembic autogenerate)."""

from app.database.models.backtest import Backtest
from app.database.models.copy_order import CopyOrder
from app.database.models.indexer_state import IndexerState
from app.database.models.market import Market
from app.database.models.portfolio import PortfolioPosition, PortfolioSnapshot
from app.database.models.position import Position
from app.database.models.ranking import Ranking
from app.database.models.risk_event import RiskEvent
from app.database.models.runtime_override import RuntimeOverride
from app.database.models.strategy import Strategy
from app.database.models.system_flag import SystemFlag
from app.database.models.trade import Trade
from app.database.models.wallet import Wallet
from app.database.models.wallet_metrics import WalletMetrics
from app.database.models.wallet_score import WalletScore

__all__ = [
    "Backtest",
    "CopyOrder",
    "IndexerState",
    "Market",
    "PortfolioPosition",
    "PortfolioSnapshot",
    "Position",
    "Ranking",
    "RiskEvent",
    "RuntimeOverride",
    "Strategy",
    "SystemFlag",
    "Trade",
    "Wallet",
    "WalletMetrics",
    "WalletScore",
]
