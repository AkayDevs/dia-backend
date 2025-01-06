"""add last_login to user model

Revision ID: 60fae0b440e4
Revises: 20240105_1000
Create Date: 2025-01-06 19:09:46.419913+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '60fae0b440e4'
down_revision: Union[str, None] = '20240105_1000'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('users', sa.Column('last_login', sa.DateTime(timezone=True), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('users', 'last_login')
    # ### end Alembic commands ### 