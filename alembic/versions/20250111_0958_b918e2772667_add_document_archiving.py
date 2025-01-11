"""add document archiving

Revision ID: b918e2772667
Revises: 9fc50cbda640
Create Date: 2024-01-11 09:58:12.345678

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b918e2772667'
down_revision: Union[str, None] = '9fc50cbda640'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create new table with desired schema
    op.create_table(
        'documents_new',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('type', sa.Enum('PDF', 'DOCX', 'XLSX', 'IMAGE', name='documenttype'), nullable=False),
        sa.Column('size', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_archived', sa.Boolean(), nullable=False, server_default=sa.text('0')),
        sa.Column('previous_version_id', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['previous_version_id'], ['documents.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Copy data from old table to new table
    op.execute('''
        INSERT INTO documents_new (
            id, name, type, size, url, user_id, 
            uploaded_at, updated_at
        )
        SELECT id, name, type, size, url, user_id, 
               uploaded_at, updated_at
        FROM documents
    ''')
    
    # Drop old table
    op.drop_table('documents')
    
    # Rename new table to original name
    op.rename_table('documents_new', 'documents')
    
    # Create indexes
    op.create_index('ix_documents_id', 'documents', ['id'])
    op.create_index('ix_documents_name', 'documents', ['name'])
    op.create_index('ix_documents_type', 'documents', ['type'])
    op.create_index('ix_documents_user_id_uploaded_at', 'documents', ['user_id', 'uploaded_at'])


def downgrade() -> None:
    # Drop new columns and constraints
    op.drop_index('ix_documents_id', 'documents')
    op.drop_index('ix_documents_name', 'documents')
    op.drop_index('ix_documents_type', 'documents')
    op.drop_index('ix_documents_user_id_uploaded_at', 'documents')
    
    # Create new table with old schema
    op.create_table(
        'documents_old',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('type', sa.Enum('PDF', 'DOCX', 'XLSX', 'IMAGE', name='documenttype'), nullable=False),
        sa.Column('size', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Copy data back
    op.execute('''
        INSERT INTO documents_old (
            id, name, type, size, url, user_id, 
            uploaded_at, updated_at
        )
        SELECT id, name, type, size, url, user_id, 
               uploaded_at, updated_at
        FROM documents
    ''')
    
    # Drop new table
    op.drop_table('documents')
    
    # Rename old table back
    op.rename_table('documents_old', 'documents') 