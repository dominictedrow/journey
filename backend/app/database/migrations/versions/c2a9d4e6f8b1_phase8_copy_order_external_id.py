"""phase 8: copy_orders.external_order_id (live-order reconciliation)

Revision ID: c2a9d4e6f8b1
Revises: b1f5c0a7e2d3
Create Date: 2026-07-04 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c2a9d4e6f8b1"
down_revision: Union[str, None] = "b1f5c0a7e2d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "copy_orders",
        sa.Column("external_order_id", sa.String(length=80), nullable=True),
    )
    op.create_index(
        op.f("ix_copy_orders_external_order_id"),
        "copy_orders",
        ["external_order_id"],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_copy_orders_external_order_id"), table_name="copy_orders")
    op.drop_column("copy_orders", "external_order_id")
