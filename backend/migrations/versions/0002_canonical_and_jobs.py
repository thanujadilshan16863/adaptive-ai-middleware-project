"""Canonical SKU/day storage and durable job status."""
from alembic import op
import sqlalchemy as sa

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('part_snapshots',
        sa.Column('run_id', sa.String(36), sa.ForeignKey('analysis_runs.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('part_id', sa.String(128), primary_key=True),
        sa.Column('part_name', sa.String(255), nullable=False), sa.Column('category', sa.String(100), nullable=False),
        sa.Column('current_stock', sa.Float(), nullable=False), sa.Column('lead_time_days', sa.Integer(), nullable=False),
        sa.Column('unit_cost', sa.Float(), nullable=False))
    op.create_table('demand_transactions',
        sa.Column('run_id', sa.String(36), primary_key=True), sa.Column('part_id', sa.String(128), primary_key=True),
        sa.Column('transaction_date', sa.Date(), primary_key=True), sa.Column('quantity_sold', sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(['run_id', 'part_id'], ['part_snapshots.run_id', 'part_snapshots.part_id'], ondelete='CASCADE'))
    op.create_table('analysis_jobs', sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.String(20), nullable=False), sa.Column('phase', sa.String(64), nullable=False),
        sa.Column('run_id', sa.String(36), sa.ForeignKey('analysis_runs.id', ondelete='SET NULL'), nullable=True),
        sa.Column('error', sa.String(1000), nullable=True))
    op.create_index('ix_analysis_jobs_status', 'analysis_jobs', ['status'])


def downgrade():
    raise RuntimeError('Destructive downgrade disabled; restore a verified backup instead')
