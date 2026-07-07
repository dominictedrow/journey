"""widen indexer_state.stream to fit per-wallet streams (history:/backfill: + address)

Revision ID: e4c9a2f7b6d1
Revises: d3b8f1c4a9e2
Create Date: 2026-07-05 13:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "e4c9a2f7b6d1"
down_revision: Union[str, None] = "d3b8f1c4a9e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "indexer_state",
        "stream",
        type_=sa.String(length=80),
        existing_type=sa.String(length=32),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "indexer_state",
        "stream",
        type_=sa.String(length=32),
        existing_type=sa.String(length=80),
        existing_nullable=False,
    )
