"""markets.neg_risk + portfolio_positions redemption columns (Module E — on-chain redemption)

Revision ID: b8e3f6a1c4d7
Revises: a7d4e9c2f1b3
Create Date: 2026-07-08 13:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b8e3f6a1c4d7"
down_revision: Union[str, None] = "a7d4e9c2f1b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "markets",
        sa.Column("neg_risk", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("portfolio_positions", sa.Column("redemption_status", sa.String(length=16)))
    op.create_index(
        op.f("ix_portfolio_positions_redemption_status"),
        "portfolio_positions",
        ["redemption_status"],
    )
    op.add_column(
        "portfolio_positions", sa.Column("redeemed_at", sa.DateTime(timezone=True))
    )
    op.add_column(
        "portfolio_positions", sa.Column("redemption_tx_hash", sa.String(length=80))
    )


def downgrade() -> None:
    op.drop_column("portfolio_positions", "redemption_tx_hash")
    op.drop_column("portfolio_positions", "redeemed_at")
    op.drop_index(
        op.f("ix_portfolio_positions_redemption_status"), table_name="portfolio_positions"
    )
    op.drop_column("portfolio_positions", "redemption_status")
    op.drop_column("markets", "neg_risk")
