"""add_updated_at_to_documents

Revision ID: 0a4de00df048
Revises: 17cc31be515d
Create Date: 2025-01-11 00:57:04.283459+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0a4de00df048'
down_revision: Union[str, None] = '17cc31be515d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('documents', sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('documents', 'updated_at')
    # ### end Alembic commands ### 