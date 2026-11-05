"""Bounded single-worker queue with durable status and explicit restart recovery."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
import uuid

from sqlalchemy import select

from backend.app.database import AnalysisJob, persist_analysis
from backend.app.pipeline import run_pipeline

logger = logging.getLogger(__name__)


class QueueFull(Exception):
    pass


class JobManager:
    def __init__(self, session_factory, engine, lock, directory, capacity):
        self.sessions, self.engine, self.lock = session_factory, engine, lock
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.slots = threading.BoundedSemaphore(capacity + 1)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='forecast-worker')
        with self.sessions.begin() as session:
            for job in session.scalars(select(AnalysisJob).where(AnalysisJob.status.in_(['queued', 'running']))):
                job.status, job.error = 'failed', 'Server restarted before completion; resubmit the file.'
                job.updated_at = datetime.now(timezone.utc)
                self._remove_upload(job.id)

    def _path(self, job_id):
        return self.directory / f'{uuid.UUID(job_id)}.upload'

    def _remove_upload(self, job_id):
        self._path(job_id).unlink(missing_ok=True)

    def submit(self, content, filename, mapping, horizon):
        if not self.slots.acquire(blocking=False):
            raise QueueFull('Forecast queue is full')
        job_id = str(uuid.uuid4())
        try:
            self._path(job_id).write_bytes(content)
            with self.sessions.begin() as session:
                session.add(AnalysisJob(id=job_id, status='queued', phase='queued'))
            self.pool.submit(self._run, job_id, filename, mapping, horizon)
            return job_id
        except Exception:
            self._remove_upload(job_id)
            self.slots.release()
            raise

    def _update(self, job_id, **values):
        with self.sessions.begin() as session:
            job = session.get(AnalysisJob, job_id)
            for name, value in values.items():
                setattr(job, name, value)
            job.updated_at = datetime.now(timezone.utc)

    def _run(self, job_id, filename, mapping, horizon):
        try:
            with self.lock:
                self._update(job_id, status='running')
                captured = []
                result = run_pipeline(self._path(job_id).read_bytes(), filename, self.engine, mapping, horizon,
                    on_canonical=captured.append, progress=lambda phase: self._update(job_id, phase=phase))
                with self.sessions.begin() as session:
                    run_id = persist_analysis(session, result, captured[0])
                    job = session.get(AnalysisJob, job_id)
                    job.status, job.phase, job.run_id = 'completed', 'completed', run_id
                    job.updated_at = datetime.now(timezone.utc)
        except Exception as exc:
            logger.exception('Forecast job %s failed', job_id)
            message = str(exc)[:900] if isinstance(exc, ValueError) else 'Inference or storage failed; see server logs.'
            self._update(job_id, status='failed', error=message)
        finally:
            self._remove_upload(job_id)
            self.slots.release()

    def close(self):
        self.pool.shutdown(wait=True)
