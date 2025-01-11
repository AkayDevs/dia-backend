"""add document retention fields

Revision ID: a7d5e0255f1f
Revises: b918e2772667
Create Date: 2025-01-11 18:09:12.123456

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7d5e0255f1f'
down_revision: Union[str, None] = 'b918e2772667'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create new table with all fields
    op.create_table(
        'documents_new',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('type', sa.Enum('pdf', 'docx', 'xlsx', 'image', name='documenttype'), nullable=False),
        sa.Column('size', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(), nullable=False),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('previous_version_id', sa.String(), nullable=True),
        sa.Column('is_archived', sa.Boolean(), nullable=False, default=False),
        sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('retention_until', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Copy data from old table to new table
    op.execute(
        """
        INSERT INTO documents_new (
            id, name, type, size, url, uploaded_at, updated_at,
            user_id, previous_version_id, is_archived, archived_at
        )
        SELECT 
            id, name, type, size, url, uploaded_at, updated_at,
            user_id, previous_version_id, is_archived, archived_at
        FROM documents
        """
    )
    
    # Drop old table and rename new table
    op.drop_table('documents')
    op.rename_table('documents_new', 'documents')
    
    # Recreate indexes
    op.create_index('ix_documents_id', 'documents', ['id'])
    op.create_index('ix_documents_name', 'documents', ['name'])
    op.create_index('ix_documents_user_id_uploaded_at', 'documents', ['user_id', 'uploaded_at'])
    op.create_index('ix_documents_type', 'documents', ['type'])


def downgrade() -> None:
    # Create old table structure
    op.create_table(
        'documents_old',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('type', sa.Enum('pdf', 'docx', 'xlsx', 'image', name='documenttype'), nullable=False),
        sa.Column('size', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(), nullable=False),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('previous_version_id', sa.String(), nullable=True),
        sa.Column('is_archived', sa.Boolean(), nullable=False, default=False),
        sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Copy data back excluding retention_until
    op.execute(
        """
        INSERT INTO documents_old (
            id, name, type, size, url, uploaded_at, updated_at,
            user_id, previous_version_id, is_archived, archived_at
        )
        SELECT 
            id, name, type, size, url, uploaded_at, updated_at,
            user_id, previous_version_id, is_archived, archived_at
        FROM documents
        """
    )
    
    # Drop new table and rename old table
    op.drop_table('documents')
    op.rename_table('documents_old', 'documents')
    
    # Recreate indexes
    op.create_index('ix_documents_id', 'documents', ['id'])
    op.create_index('ix_documents_name', 'documents', ['name'])
    op.create_index('ix_documents_user_id_uploaded_at', 'documents', ['user_id', 'uploaded_at'])
    op.create_index('ix_documents_type', 'documents', ['type']) 