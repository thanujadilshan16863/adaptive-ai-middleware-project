"""Explicit migration, online SQLite backup and opt-in run retention."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

from sqlalchemy import delete, select
from backend.app.database import AnalysisRun, AnalysisJob, DemandTransaction, PartSnapshot, Session, engine, migrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['migrate', 'backup', 'retention'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--apply', action='store_true', help='Actually delete old runs; default is a count only')
    args = parser.parse_args()
    if args.action == 'migrate':
        migrate(engine)
        print('Database migrated to revision 0002')
    elif args.action == 'backup':
        if engine.dialect.name != 'sqlite' or not args.output:
            parser.error('SQLite backup requires --output; use pg_dump for PostgreSQL')
        source = Path(engine.url.database).resolve()
        target = args.output.resolve()
        if target == source or target.exists():
            parser.error('Backup target must be a new file distinct from the source')
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(source) as origin, sqlite3.connect(target) as destination:
            origin.backup(destination)
        print(f'Consistent SQLite backup created: {target}')
    else:
        if args.days < 1:
            parser.error('--days must be positive')
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
        with Session.begin() as session:
            ids = list(session.scalars(select(AnalysisRun.id).where(AnalysisRun.created_at < cutoff)))
            if args.apply and ids:
                session.execute(delete(AnalysisJob).where(AnalysisJob.run_id.in_(ids)))
                session.execute(delete(DemandTransaction).where(DemandTransaction.run_id.in_(ids)))
                session.execute(delete(PartSnapshot).where(PartSnapshot.run_id.in_(ids)))
                session.execute(delete(AnalysisRun).where(AnalysisRun.id.in_(ids)))
            print(f'{len(ids)} old runs: {"deleted" if args.apply else "dry run; use --apply after backup"}')


if __name__ == '__main__':
    main()
