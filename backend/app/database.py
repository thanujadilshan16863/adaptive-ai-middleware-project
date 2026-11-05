from datetime import datetime, timezone
import uuid

from sqlalchemy import JSON, Date, DateTime, Float, Integer, String, ForeignKey, ForeignKeyConstraint, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from backend.app.config import settings


class Base(DeclarativeBase):
    pass


class AnalysisRun(Base):
    __tablename__ = 'analysis_runs'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    result: Mapped[dict] = mapped_column(JSON)


class PartSnapshot(Base):
    __tablename__ = 'part_snapshots'
    run_id: Mapped[str] = mapped_column(ForeignKey('analysis_runs.id', ondelete='CASCADE'), primary_key=True)
    part_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    part_name: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(100))
    current_stock: Mapped[float] = mapped_column(Float)
    lead_time_days: Mapped[int] = mapped_column(Integer)
    unit_cost: Mapped[float] = mapped_column(Float)


class DemandTransaction(Base):
    __tablename__ = 'demand_transactions'
    __table_args__ = (ForeignKeyConstraint(['run_id', 'part_id'], ['part_snapshots.run_id', 'part_snapshots.part_id'], ondelete='CASCADE'),)
    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    part_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    transaction_date: Mapped[datetime] = mapped_column(Date, primary_key=True)
    quantity_sold: Mapped[float] = mapped_column(Float)


class AnalysisJob(Base):
    __tablename__ = 'analysis_jobs'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    status: Mapped[str] = mapped_column(String(20), default='queued', index=True)
    phase: Mapped[str] = mapped_column(String(64), default='queued')
    run_id: Mapped[str | None] = mapped_column(ForeignKey('analysis_runs.id', ondelete='SET NULL'), nullable=True)
    error: Mapped[str | None] = mapped_column(String(1000), nullable=True)


engine = create_engine(settings.database_url, pool_pre_ping=True,
                       connect_args={'check_same_thread': False} if settings.database_url.startswith('sqlite') else {})
Session = sessionmaker(engine)

if engine.dialect.name == 'sqlite':
    @event.listens_for(engine, 'connect')
    def sqlite_settings(connection, record):
        cursor = connection.cursor()
        cursor.execute('PRAGMA foreign_keys=ON')
        cursor.execute('PRAGMA busy_timeout=30000')
        cursor.close()


def migrate(target_engine):
    from pathlib import Path
    from alembic import command
    from alembic.config import Config
    root = Path(__file__).resolve().parents[1]
    config = Config()
    config.set_main_option('script_location', str(root / 'migrations'))
    with target_engine.begin() as connection:
        config.attributes['connection'] = connection
        command.upgrade(config, 'head')


def persist_analysis(session, result, canonical):
    """Atomically persist immutable run, SKU snapshot and canonical daily sales."""
    from sqlalchemy import insert
    run = AnalysisRun(result=result)
    session.add(run)
    session.flush()
    latest = canonical.groupby('part_id', sort=True).tail(1)
    parts = [{'run_id': run.id, 'part_id': str(row.part_id), 'part_name': str(row.part_name),
              'category': str(row.category), 'current_stock': float(row.current_stock),
              'lead_time_days': int(row.lead_time_days), 'unit_cost': float(row.unit_cost)}
             for row in latest.itertuples()]
    session.execute(insert(PartSnapshot), parts)
    daily = canonical.groupby(['part_id', 'transaction_date'], as_index=False).quantity_sold.sum()
    records = [{'run_id': run.id, 'part_id': str(row.part_id), 'transaction_date': row.transaction_date.date(),
                'quantity_sold': float(row.quantity_sold)} for row in daily.itertuples()]
    for start in range(0, len(records), 2000):
        session.execute(insert(DemandTransaction), records[start:start + 2000])
    return run.id
