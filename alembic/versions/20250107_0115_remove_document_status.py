"""remove document status

Revision ID: 20250107_0115
Revises: 20250107_0114
Create Date: 2024-01-07 01:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20250107_0115'
down_revision: Union[str, None] = '20250107_0114'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the status-related index first
    op.drop_index('ix_documents_status_type', table_name='documents')
    
    # Drop the status column
    op.drop_column('documents', 'status')


def downgrade() -> None:
    # Add back the status column
    op.add_column('documents', sa.Column(
        'status',
        sa.Enum('pending', 'processing', 'completed', 'failed', name='analysisstatus'),
        nullable=False,
        server_default='pending'
    ))
    
    # Recreate the index
    op.create_index(
        'ix_documents_status_type',
        'documents',
        ['status', 'type'],
        unique=False
    ) 