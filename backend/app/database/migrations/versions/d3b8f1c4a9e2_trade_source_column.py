"""trades.source (chain | data_api ingestion provenance)

Revision ID: d3b8f1c4a9e2
Revises: c2a9d4e6f8b1
Create Date: 2026-07-05 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "d3b8f1c4a9e2"
down_revision: Union[str, None] = "c2a9d4e6f8b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # server_default 'chain' backfills existing on-chain rows; new inserts set it explicitly.
    op.add_column(
        "trades",
        sa.Column("source", sa.String(length=16), nullable=False, server_default="chain"),
    )


def downgrade() -> None:
    op.drop_column("trades", "source")
