"""Preserve legacy result-only databases and initialize new ones."""
from alembic import op
import sqlalchemy as sa

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    if not sa.inspect(op.get_bind()).has_table('analysis_runs'):
        op.create_table('analysis_runs', sa.Column('id', sa.String(36), primary_key=True),
                        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
                        sa.Column('result', sa.JSON(), nullable=False))


def downgrade():
    raise RuntimeError('Destructive downgrade disabled; restore a verified backup instead')
