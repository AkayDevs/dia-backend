"""add updated_at

Revision ID: 20250107_0118
Revises: 20250107_0117
Create Date: 2024-01-07 01:18:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20250107_0118'
down_revision: Union[str, None] = '20250107_0117'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create a new temporary table with all columns including updated_at
    op.execute("""
        CREATE TABLE documents_new (
            id VARCHAR NOT NULL, 
            name VARCHAR NOT NULL,
            type VARCHAR NOT NULL,
            uploaded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            size INTEGER NOT NULL,
            url VARCHAR NOT NULL,
            user_id VARCHAR NOT NULL,
            file_hash VARCHAR(64) NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    
    # Copy data from the old table to the new table
    op.execute("""
        INSERT INTO documents_new 
        SELECT 
            id, 
            name,
            type,
            uploaded_at,
            uploaded_at as updated_at,
            size,
            url,
            user_id,
            file_hash
        FROM documents
    """)
    
    # Drop the old indexes
    op.drop_index('ix_documents_user_id_file_hash')
    op.drop_index('ix_documents_file_hash')
    op.drop_index('ix_documents_user_id_uploaded_at')
    op.drop_index('ix_documents_type')
    op.drop_index('ix_documents_name')
    op.drop_index('ix_documents_id')
    
    # Drop the old table
    op.drop_table('documents')
    
    # Rename the new table to the original name
    op.execute('ALTER TABLE documents_new RENAME TO documents')
    
    # Create trigger for auto-updating updated_at
    op.execute("""
        CREATE TRIGGER update_documents_updated_at
            AFTER UPDATE ON documents
            FOR EACH ROW
            WHEN NEW.updated_at = OLD.updated_at
        BEGIN
            UPDATE documents SET updated_at = CURRENT_TIMESTAMP
            WHERE id = NEW.id;
        END;
    """)
    
    # Recreate all indexes
    op.create_index('ix_documents_id', 'documents', ['id'])
    op.create_index('ix_documents_name', 'documents', ['name'])
    op.create_index('ix_documents_type', 'documents', ['type'])
    op.create_index(
        'ix_documents_user_id_uploaded_at',
        'documents',
        ['user_id', sa.text('uploaded_at DESC')]
    )
    op.create_index('ix_documents_file_hash', 'documents', ['file_hash'])
    op.create_index(
        'ix_documents_user_id_file_hash',
        'documents',
        ['user_id', 'file_hash'],
        unique=True
    )


def downgrade() -> None:
    # Drop the trigger first
    op.execute("DROP TRIGGER IF EXISTS update_documents_updated_at")
    
    # Create a new temporary table without updated_at
    op.execute("""
        CREATE TABLE documents_new (
            id VARCHAR NOT NULL, 
            name VARCHAR NOT NULL,
            type VARCHAR NOT NULL,
            uploaded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            size INTEGER NOT NULL,
            url VARCHAR NOT NULL,
            user_id VARCHAR NOT NULL,
            file_hash VARCHAR(64) NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    
    # Copy data from the current table to the new table
    op.execute("""
        INSERT INTO documents_new 
        SELECT 
            id, 
            name,
            type,
            uploaded_at,
            size,
            url,
            user_id,
            file_hash
        FROM documents
    """)
    
    # Drop all indexes
    op.drop_index('ix_documents_user_id_file_hash')
    op.drop_index('ix_documents_file_hash')
    op.drop_index('ix_documents_user_id_uploaded_at')
    op.drop_index('ix_documents_type')
    op.drop_index('ix_documents_name')
    op.drop_index('ix_documents_id')
    
    # Drop the current table
    op.drop_table('documents')
    
    # Rename the new table to the original name
    op.execute('ALTER TABLE documents_new RENAME TO documents')
    
    # Recreate all indexes
    op.create_index('ix_documents_id', 'documents', ['id'])
    op.create_index('ix_documents_name', 'documents', ['name'])
    op.create_index('ix_documents_type', 'documents', ['type'])
    op.create_index(
        'ix_documents_user_id_uploaded_at',
        'documents',
        ['user_id', sa.text('uploaded_at DESC')]
    )
    op.create_index('ix_documents_file_hash', 'documents', ['file_hash'])
    op.create_index(
        'ix_documents_user_id_file_hash',
        'documents',
        ['user_id', 'file_hash'],
        unique=True
    ) 