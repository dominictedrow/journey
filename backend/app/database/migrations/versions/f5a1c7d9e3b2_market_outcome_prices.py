"""markets.outcome_prices (Gamma resolution payouts for settlement)

Revision ID: f5a1c7d9e3b2
Revises: e4c9a2f7b6d1
Create Date: 2026-07-07 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f5a1c7d9e3b2"
down_revision: Union[str, None] = "e4c9a2f7b6d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Parallel to markets.outcomes/token_ids: Gamma's outcomePrices list ("1"/"0" once
    # resolved), aligned by index with token_ids. Null until the market resolves.
    op.add_column(
        "markets",
        sa.Column("outcome_prices", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("markets", "outcome_prices")
