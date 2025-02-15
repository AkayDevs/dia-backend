"""fix_analysis_models_schema

Revision ID: ad603fe11c61
Revises: f0fbc78ad602
Create Date: 2025-02-15 07:19:44.264560

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ad603fe11c61'
down_revision: Union[str, None] = 'f0fbc78ad602'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop temporary tables if they exist
    op.drop_table('algorithms_new', if_exists=True)
    op.drop_table('analysis_steps_new', if_exists=True)
    op.drop_table('analysis_types_new', if_exists=True)
    
    # Add missing constraints using batch operations
    with op.batch_alter_table('algorithms', schema=None) as batch_op:
        batch_op.create_unique_constraint('uix_algorithm_step_code_version', ['step_id', 'code', 'version'])
    
    with op.batch_alter_table('analysis_steps', schema=None) as batch_op:
        # Add missing columns with default values
        batch_op.add_column(sa.Column('code', sa.String(100), nullable=True))
        batch_op.add_column(sa.Column('version', sa.String(20), nullable=True))
        batch_op.add_column(sa.Column('result_schema', sa.String(255), nullable=True))
        batch_op.add_column(sa.Column('implementation_path', sa.String(255), nullable=True))
        batch_op.add_column(sa.Column('is_active', sa.Boolean(), server_default='1', nullable=False))
        
        # Update column type
        batch_op.alter_column('name',
                            existing_type=sa.VARCHAR(length=27),
                            type_=sa.String(length=100),
                            existing_nullable=False)
        
        # Create unique constraint
        batch_op.create_unique_constraint('uix_step_analysis_code_version', ['analysis_type_id', 'code', 'version'])
        
        # Make columns non-nullable after adding them
        batch_op.alter_column('code', nullable=False)
        batch_op.alter_column('version', nullable=False)
        batch_op.alter_column('result_schema', nullable=False)
        batch_op.alter_column('implementation_path', nullable=False)
    
    with op.batch_alter_table('analysis_types', schema=None) as batch_op:
        # Update column types and constraints
        batch_op.alter_column('name',
                            existing_type=sa.VARCHAR(length=19),
                            type_=sa.String(length=100),
                            existing_nullable=False)
        
        batch_op.alter_column('code', nullable=False)
        batch_op.alter_column('version', nullable=False)
        batch_op.alter_column('implementation_path', nullable=False)
        
        # Add unique constraints
        batch_op.create_unique_constraint('uix_analysis_type_code_version', ['code', 'version'])
        batch_op.create_unique_constraint('uq_analysis_types_code', ['code'])
    
    # Remove unnecessary indexes
    with op.batch_alter_table('analyses', schema=None) as batch_op:
        batch_op.drop_index('ix_analyses_analysis_type_id')
        batch_op.drop_index('ix_analyses_created_at')
        batch_op.drop_index('ix_analyses_document_id')
        batch_op.drop_index('ix_analyses_mode_status')
        batch_op.drop_index('ix_analyses_status')
    
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.drop_index('ix_documents_archived_status')
        batch_op.drop_index('ix_documents_version')


def downgrade() -> None:
    # Remove constraints
    with op.batch_alter_table('algorithms', schema=None) as batch_op:
        batch_op.drop_constraint('uix_algorithm_step_code_version', type_='unique')
    
    with op.batch_alter_table('analysis_steps', schema=None) as batch_op:
        batch_op.drop_constraint('uix_step_analysis_code_version', type_='unique')
        
        # Revert column type
        batch_op.alter_column('name',
                            existing_type=sa.String(length=100),
                            type_=sa.VARCHAR(length=27),
                            existing_nullable=False)
        
        # Drop added columns
        batch_op.drop_column('is_active')
        batch_op.drop_column('implementation_path')
        batch_op.drop_column('result_schema')
        batch_op.drop_column('version')
        batch_op.drop_column('code')
    
    with op.batch_alter_table('analysis_types', schema=None) as batch_op:
        batch_op.drop_constraint('uq_analysis_types_code', type_='unique')
        batch_op.drop_constraint('uix_analysis_type_code_version', type_='unique')
        
        # Revert column type
        batch_op.alter_column('name',
                            existing_type=sa.String(length=100),
                            type_=sa.VARCHAR(length=19),
                            existing_nullable=False)
        
        # Make columns nullable again
        batch_op.alter_column('code', nullable=True)
        batch_op.alter_column('version', nullable=True)
        batch_op.alter_column('implementation_path', nullable=True)
    
    # Recreate indexes
    with op.batch_alter_table('analyses', schema=None) as batch_op:
        batch_op.create_index('ix_analyses_status', ['status'])
        batch_op.create_index('ix_analyses_mode_status', ['mode', 'status'])
        batch_op.create_index('ix_analyses_document_id', ['document_id'])
        batch_op.create_index('ix_analyses_created_at', ['created_at'])
        batch_op.create_index('ix_analyses_analysis_type_id', ['analysis_type_id'])
    
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.create_index('ix_documents_version', ['previous_version_id'])
        batch_op.create_index('ix_documents_archived_status', ['is_archived', 'archived_at', 'retention_until']) 