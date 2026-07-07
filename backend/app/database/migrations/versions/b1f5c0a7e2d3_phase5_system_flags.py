"""phase 5: system_flags (kill switch)

Revision ID: b1f5c0a7e2d3
Revises: a64186020815
Create Date: 2026-07-03 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b1f5c0a7e2d3"
down_revision: Union[str, None] = "a64186020815"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "system_flags",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_system_flags")),
    )


def downgrade() -> None:
    op.drop_table("system_flags")
