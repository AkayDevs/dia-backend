"""update document indexes

Revision ID: 045c395d27d4
Revises: 134cf601cb78
Create Date: 2025-01-14 07:02:23.123456

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '045c395d27d4'
down_revision = '134cf601cb78'
branch_labels = None
depends_on = None


def upgrade():
    # Create index for archived documents
    op.create_index(
        'ix_documents_archived_status',
        'documents',
        ['is_archived', 'archived_at', 'retention_until']
    )
    
    # Create index for document versions
    op.create_index(
        'ix_documents_version',
        'documents',
        ['previous_version_id']
    )


def downgrade():
    # Drop the new indexes
    op.drop_index('ix_documents_archived_status', table_name='documents')
    op.drop_index('ix_documents_version', table_name='documents') 