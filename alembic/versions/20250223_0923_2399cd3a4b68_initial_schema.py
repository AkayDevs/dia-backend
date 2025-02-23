"""initial_schema

Revision ID: 2399cd3a4b68
Revises: 
Create Date: 2025-02-23 09:23:59.126623+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite


# revision identifiers, used by Alembic.
revision: str = '2399cd3a4b68'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('email', sa.String, unique=True, nullable=False),
        sa.Column('name', sa.String, nullable=False),
        sa.Column('hashed_password', sa.String, nullable=False),
        sa.Column('avatar', sa.String),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column('last_login', sa.DateTime(timezone=True)),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default='1'),
        sa.Column('is_verified', sa.Boolean, nullable=False, server_default='0'),
        sa.Column('verification_token', sa.String),
        sa.Column('password_reset_token', sa.String),
        sa.Column('password_reset_expires', sa.DateTime(timezone=True))
    )

    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('type', sa.String(20), nullable=False),
        sa.Column('size', sa.Integer, nullable=False),
        sa.Column('url', sa.String, nullable=False),
        sa.Column('previous_version_id', sa.String),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('is_archived', sa.Boolean, nullable=False, server_default='0'),
        sa.Column('archived_at', sa.DateTime(timezone=True)),
        sa.Column('retention_until', sa.DateTime(timezone=True))
    )

    # Create tags table
    op.create_table(
        'tags',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String, nullable=False, unique=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    # Create document_tags association table
    op.create_table(
        'document_tags',
        sa.Column('document_id', sa.String(36), sa.ForeignKey('documents.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('tag_id', sa.Integer, sa.ForeignKey('tags.id', ondelete='CASCADE'), primary_key=True)
    )

    # Create analysis_definitions table
    op.create_table(
        'analysis_definitions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('code', sa.String(100), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('version', sa.String(20), nullable=False),
        sa.Column('description', sa.String(500)),
        sa.Column('supported_document_types', sqlite.JSON),
        sa.Column('implementation_path', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.UniqueConstraint('code', 'version', name='uix_analysis_type_code_version')
    )

    # Create step_definitions table
    op.create_table(
        'step_definitions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('code', sa.String(100), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('version', sa.String(20), nullable=False),
        sa.Column('description', sa.String(500)),
        sa.Column('order', sa.Integer, nullable=False),
        sa.Column('analysis_definition_id', sa.String(36), sa.ForeignKey('analysis_definitions.id'), nullable=False),
        sa.Column('result_schema_path', sa.String(255), nullable=False),
        sa.Column('base_parameters', sqlite.JSON),
        sa.Column('implementation_path', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.UniqueConstraint('analysis_definition_id', 'code', 'version', name='uix_step_analysis_code_version')
    )

    # Create algorithm_definitions table
    op.create_table(
        'algorithm_definitions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('code', sa.String(100), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.String(500)),
        sa.Column('version', sa.String(20), nullable=False),
        sa.Column('step_id', sa.String(36), sa.ForeignKey('step_definitions.id'), nullable=False),
        sa.Column('supported_document_types', sqlite.JSON),
        sa.Column('parameters', sqlite.JSON),
        sa.Column('implementation_path', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.UniqueConstraint('step_id', 'code', 'version', name='uix_algorithm_step_code_version')
    )

    # Create analysis_runs table
    op.create_table(
        'analysis_runs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('document_id', sa.String(36), sa.ForeignKey('documents.id'), nullable=False),
        sa.Column('analysis_definition_id', sa.String(36), sa.ForeignKey('analysis_definitions.id'), nullable=False),
        sa.Column('mode', sa.String(20), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('config', sqlite.JSON, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('error_message', sa.String(500))
    )

    # Create step_execution_results table
    op.create_table(
        'step_execution_results',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('analysis_run_id', sa.String(36), sa.ForeignKey('analysis_runs.id'), nullable=False),
        sa.Column('step_definition_id', sa.String(36), sa.ForeignKey('step_definitions.id'), nullable=False),
        sa.Column('algorithm_definition_id', sa.String(36), sa.ForeignKey('algorithm_definitions.id'), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('parameters', sqlite.JSON),
        sa.Column('result', sqlite.JSON),
        sa.Column('user_corrections', sqlite.JSON),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('error_message', sa.String(500))
    )

    # Create indexes
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_verification', 'users', ['verification_token', 'is_verified'])
    op.create_index('ix_users_password_reset', 'users', ['password_reset_token', 'password_reset_expires'])
    op.create_index('ix_users_role_active', 'users', ['role', 'is_active'])

    op.create_index('ix_documents_name', 'documents', ['name'])
    op.create_index('ix_documents_user_id_uploaded_at', 'documents', ['user_id', 'uploaded_at'])
    op.create_index('ix_documents_type', 'documents', ['type'])

    op.create_index('ix_document_tags_document_id', 'document_tags', ['document_id'])
    op.create_index('ix_document_tags_tag_id', 'document_tags', ['tag_id'])

    op.create_index('ix_analysis_runs_document_id', 'analysis_runs', ['document_id'])
    op.create_index('ix_analysis_runs_analysis_definition_id', 'analysis_runs', ['analysis_definition_id'])
    op.create_index('ix_analysis_runs_status', 'analysis_runs', ['status'])

    op.create_index('ix_step_execution_results_analysis_run_id', 'step_execution_results', ['analysis_run_id'])
    op.create_index('ix_step_execution_results_step_definition_id', 'step_execution_results', ['step_definition_id'])
    op.create_index('ix_step_execution_results_algorithm_definition_id', 'step_execution_results', ['algorithm_definition_id'])
    op.create_index('ix_step_execution_results_status', 'step_execution_results', ['status'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('step_execution_results')
    op.drop_table('analysis_runs')
    op.drop_table('algorithm_definitions')
    op.drop_table('step_definitions')
    op.drop_table('analysis_definitions')
    op.drop_table('document_tags')
    op.drop_table('tags')
    op.drop_table('documents')
    op.drop_table('users') 