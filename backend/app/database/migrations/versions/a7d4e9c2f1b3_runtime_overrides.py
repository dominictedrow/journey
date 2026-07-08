"""runtime_overrides table (Module D — activity governor's runtime-mutable settings layer)

Revision ID: a7d4e9c2f1b3
Revises: f5a1c7d9e3b2
Create Date: 2026-07-08 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a7d4e9c2f1b3"
down_revision: Union[str, None] = "f5a1c7d9e3b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "runtime_overrides",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("reason", sa.String(length=255), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_runtime_overrides")),
    )


def downgrade() -> None:
    op.drop_table("runtime_overrides")
