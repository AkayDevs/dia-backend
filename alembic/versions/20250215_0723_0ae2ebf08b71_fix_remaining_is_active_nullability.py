"""fix_remaining_is_active_nullability

Revision ID: 0ae2ebf08b71
Revises: 0e80a9ff2730
Create Date: 2025-02-15 07:23:44.264560

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0ae2ebf08b71'
down_revision = '0e80a9ff2730'
branch_labels = None
depends_on = None

def upgrade():
    # Fix algorithms table
    with op.batch_alter_table('algorithms', schema=None) as batch_op:
        # First set any NULL values to True
        op.execute("UPDATE algorithms SET is_active = 1 WHERE is_active IS NULL")
        
        # Then make the column non-nullable with a default value
        batch_op.alter_column('is_active',
                            existing_type=sa.Boolean(),
                            nullable=False,
                            server_default='1')
    
    # Fix analysis_types table
    with op.batch_alter_table('analysis_types', schema=None) as batch_op:
        # First set any NULL values to True
        op.execute("UPDATE analysis_types SET is_active = 1 WHERE is_active IS NULL")
        
        # Then make the column non-nullable with a default value
        batch_op.alter_column('is_active',
                            existing_type=sa.Boolean(),
                            nullable=False,
                            server_default='1')

def downgrade():
    # Revert algorithms table
    with op.batch_alter_table('algorithms', schema=None) as batch_op:
        batch_op.alter_column('is_active',
                            existing_type=sa.Boolean(),
                            nullable=True,
                            server_default=None)
    
    # Revert analysis_types table
    with op.batch_alter_table('analysis_types', schema=None) as batch_op:
        batch_op.alter_column('is_active',
                            existing_type=sa.Boolean(),
                            nullable=True,
                            server_default=None) 