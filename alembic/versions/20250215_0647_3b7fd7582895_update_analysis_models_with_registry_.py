"""update_analysis_models_with_registry_system

Revision ID: 3b7fd7582895
Revises: 6a64e9f67fb9
Create Date: 2025-02-15 06:47:28.072164+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3b7fd7582895'
down_revision: Union[str, None] = '6a64e9f67fb9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass 