"""update analysis result model with granular fields

Revision ID: 470047cbbbd3
Revises: a7d5e0255f1f
Create Date: 2025-01-13 12:49:23.123456

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from app.enums.analysis import AnalysisStatus, AnalysisMode


# revision identifiers, used by Alembic.
revision = '470047cbbbd3'
down_revision = 'a7d5e0255f1f'
branch_labels = None
depends_on = None


def upgrade():
    # Create new table with all fields
    op.create_table(
        'analysis_results_new',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('document_id', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('status', sa.Enum(AnalysisStatus), nullable=False),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('error', sa.String(), nullable=True),
        sa.Column('progress', sa.Float(), nullable=False),
        sa.Column('status_message', sa.String(), nullable=True),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('mode', sa.Enum(AnalysisMode), nullable=True),
        sa.Column('current_step', sa.String(), nullable=True),
        sa.Column('step_results', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes on the new table
    op.create_index('ix_analysis_results_new_id', 'analysis_results_new', ['id'])
    op.create_index('ix_analysis_results_new_document_id', 'analysis_results_new', ['document_id'])
    op.create_index('ix_analysis_results_new_type', 'analysis_results_new', ['type'])
    op.create_index('ix_analysis_results_new_status', 'analysis_results_new', ['status'])
    op.create_index('ix_analysis_results_new_created_at', 'analysis_results_new', ['created_at'])
    op.create_index('ix_analysis_results_new_mode', 'analysis_results_new', ['mode'])
    op.create_index('ix_analysis_results_new_current_step', 'analysis_results_new', ['current_step'])
    op.create_index('ix_analysis_results_new_mode_status', 'analysis_results_new', ['mode', 'status'])
    op.create_index('ix_analysis_results_new_document_id_type', 'analysis_results_new', ['document_id', 'type'])

    # Copy data from old table to new table
    op.execute("""
        INSERT INTO analysis_results_new (
            id, document_id, type, status, result, error,
            progress, status_message, parameters, created_at,
            completed_at
        )
        SELECT id, document_id, type, status, result, error,
               progress, status_message, parameters, created_at,
               completed_at
        FROM analysis_results;
    """)

    # Drop old table and rename new table
    op.drop_table('analysis_results')
    op.rename_table('analysis_results_new', 'analysis_results')


def downgrade():
    # Create old table structure
    op.create_table(
        'analysis_results_old',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('document_id', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('error', sa.String(), nullable=True),
        sa.Column('progress', sa.Float(), nullable=False),
        sa.Column('status_message', sa.String(), nullable=True),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Copy data back (excluding new fields)
    op.execute("""
        INSERT INTO analysis_results_old (
            id, document_id, type, status, result, error,
            progress, status_message, parameters, created_at,
            completed_at
        )
        SELECT id, document_id, type, status, result, error,
               progress, status_message, parameters, created_at,
               completed_at
        FROM analysis_results;
    """)

    # Drop new table and rename old table back
    op.drop_table('analysis_results')
    op.rename_table('analysis_results_old', 'analysis_results')

    # Recreate original indexes
    op.create_index('ix_analysis_results_id', 'analysis_results', ['id'])
    op.create_index('ix_analysis_results_document_id', 'analysis_results', ['document_id'])
    op.create_index('ix_analysis_results_type', 'analysis_results', ['type'])
    op.create_index('ix_analysis_results_status', 'analysis_results', ['status'])
    op.create_index('ix_analysis_results_created_at', 'analysis_results', ['created_at'])
    op.create_index('ix_analysis_results_document_id_type', 'analysis_results', ['document_id', 'type']) 