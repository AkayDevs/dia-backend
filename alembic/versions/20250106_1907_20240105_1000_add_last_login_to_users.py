"""add last_login to users

Revision ID: 20240105_1000
Revises: a8ec14b9de70
Create Date: 2025-01-06 19:07:53.933489+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20240105_1000'
down_revision: Union[str, None] = 'a8ec14b9de70'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass 