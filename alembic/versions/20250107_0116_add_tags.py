"""add tags

Revision ID: 20250107_0116
Revises: 20250107_0115
Create Date: 2024-01-07 01:16:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20250107_0116'
down_revision: Union[str, None] = '20250107_0115'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create tags table
    op.create_table(
        'tags',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes on tags table
    op.create_index('ix_tags_id', 'tags', ['id'])
    op.create_index('ix_tags_name', 'tags', ['name'])
    
    # Create document_tags association table
    op.create_table(
        'document_tags',
        sa.Column('document_id', sa.String(), nullable=False),
        sa.Column('tag_id', sa.String(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['tag_id'], ['tags.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('document_id', 'tag_id')
    )
    
    # Create indexes on document_tags table
    op.create_index('ix_document_tags_document_id', 'document_tags', ['document_id'])
    op.create_index('ix_document_tags_tag_id', 'document_tags', ['tag_id'])
    
    # Create default "untagged" tag
    op.execute(
        """
        INSERT INTO tags (id, name, created_at)
        VALUES (
            '00000000-0000-0000-0000-000000000000',
            'untagged',
            CURRENT_TIMESTAMP
        )
        """
    )


def downgrade() -> None:
    # Drop document_tags table and its indexes
    op.drop_index('ix_document_tags_tag_id', table_name='document_tags')
    op.drop_index('ix_document_tags_document_id', table_name='document_tags')
    op.drop_table('document_tags')
    
    # Drop tags table and its indexes
    op.drop_index('ix_tags_name', table_name='tags')
    op.drop_index('ix_tags_id', table_name='tags')
    op.drop_table('tags') 