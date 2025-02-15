"""fix_analysis_steps_is_active_nullable

Revision ID: 0e80a9ff2730
Revises: ad603fe11c61
Create Date: 2025-02-15 07:21:44.264560

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0e80a9ff2730'
down_revision = 'ad603fe11c61'
branch_labels = None
depends_on = None

def upgrade():
    # Use batch operations for SQLite compatibility
    with op.batch_alter_table('analysis_steps', schema=None) as batch_op:
        # First set any NULL values to True
        op.execute("UPDATE analysis_steps SET is_active = 1 WHERE is_active IS NULL")
        
        # Then make the column non-nullable with a default value
        batch_op.alter_column('is_active',
                            existing_type=sa.Boolean(),
                            nullable=False,
                            server_default='1')

def downgrade():
    # Use batch operations for SQLite compatibility
    with op.batch_alter_table('analysis_steps', schema=None) as batch_op:
        # Make the column nullable again
        batch_op.alter_column('is_active',
                            existing_type=sa.Boolean(),
                            nullable=True,
                            server_default=None) 