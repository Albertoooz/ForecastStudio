"""Add DataSource status, last_error, query_or_table.

Revision ID: 20260407_01
Revises:
Create Date: 2026-04-07
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260407_01"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "data_sources",
        sa.Column("query_or_table", sa.Text(), nullable=True),
    )
    op.add_column(
        "data_sources",
        sa.Column("status", sa.String(20), server_default="connected", nullable=False),
    )
    op.add_column(
        "data_sources",
        sa.Column("last_error", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("data_sources", "last_error")
    op.drop_column("data_sources", "status")
    op.drop_column("data_sources", "query_or_table")
