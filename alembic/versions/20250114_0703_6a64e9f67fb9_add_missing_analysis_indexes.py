"""add missing analysis indexes

Revision ID: 6a64e9f67fb9
Revises: 045c395d27d4
Create Date: 2025-01-14 07:03:23.123456

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6a64e9f67fb9'
down_revision = '045c395d27d4'
branch_labels = None
depends_on = None


def upgrade():
    # Add missing indexes for analyses table
    op.create_index('ix_analyses_document_id', 'analyses', ['document_id'])
    op.create_index('ix_analyses_analysis_type_id', 'analyses', ['analysis_type_id'])
    
    # Add additional useful indexes
    op.create_index('ix_analyses_status', 'analyses', ['status'])
    op.create_index('ix_analyses_mode_status', 'analyses', ['mode', 'status'])
    op.create_index('ix_analyses_created_at', 'analyses', ['created_at'])


def downgrade():
    # Drop all the new indexes
    op.drop_index('ix_analyses_document_id', table_name='analyses')
    op.drop_index('ix_analyses_analysis_type_id', table_name='analyses')
    op.drop_index('ix_analyses_status', table_name='analyses')
    op.drop_index('ix_analyses_mode_status', table_name='analyses')
    op.drop_index('ix_analyses_created_at', table_name='analyses') 