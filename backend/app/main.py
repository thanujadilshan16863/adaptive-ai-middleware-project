import hmac
import json
import logging
import threading
import time
import uuid
import socket
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select, func
from starlette.concurrency import run_in_threadpool

from backend.app.config import settings
from backend.app.database import AnalysisRun, AnalysisJob, DemandTransaction, PartSnapshot, Session, engine as db_engine, migrate, persist_analysis
from backend.app.jobs import JobManager, QueueFull
from backend.app.pipeline import run_pipeline
from backend.app.schemas import SchemaMappingResult
from backend.app.services.phase1_ingestion import ingest, profile
from backend.app.services.phase2_llm_mapper import HybridProfileLLMMapper
from backend.app.services.phase4_forecaster import build_engine

logger = logging.getLogger(__name__)
forecast_engine = build_engine()
job_lock = threading.Lock()
schema_slots = threading.BoundedSemaphore(2)


@asynccontextmanager
async def lifespan(app):
    if settings.deployment_mode == 'production' and len(settings.api_key) < 32:
        raise RuntimeError('Production mode requires an API_KEY of at least 32 characters')
    migrate(db_engine)
    app.state.jobs = JobManager(Session, forecast_engine, job_lock, settings.job_directory, settings.job_queue_size)
    try:
        yield
    finally:
        await run_in_threadpool(app.state.jobs.close)


app = FastAPI(title='Adaptive Demand Forecasting Middleware', version='2.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type', 'X-API-Key'])


@app.middleware('http')
async def request_observability(request: Request, call_next):
    request_id = str(uuid.uuid4())
    size = request.headers.get('content-length')
    if size and (not size.isdigit() or int(size) > (settings.max_upload_mb + 1) * 1024**2):
        return JSONResponse({'detail': 'Request exceeds upload limit'}, status_code=413)
    start = time.perf_counter()
    response = await call_next(request)
    response.headers['X-Request-ID'] = request_id
    response.headers['X-Content-Type-Options'] = 'nosniff'
    logger.info('request_id=%s method=%s path=%s status=%s elapsed_ms=%.1f',
                request_id, request.method, request.url.path, response.status_code, (time.perf_counter()-start)*1000)
    return response


def authenticate(x_api_key: str = Header(default='')):
    if settings.api_key and not hmac.compare_digest(settings.api_key, x_api_key):
        raise HTTPException(401, 'Invalid API key')


async def read_upload(file):
    try:
        data = await file.read(settings.max_upload_mb * 1024**2 + 1)
    finally:
        await file.close()
    if len(data) > settings.max_upload_mb * 1024**2:
        raise HTTPException(413, 'Upload too large')
    return data


@app.get('/health')
def health():
    return {'status': 'ok', 'forecast_mode': settings.forecast_mode,
            'forecast_strategy': settings.forecast_strategy,
            'chronos_loaded': forecast_engine.pipeline is not None,
            'llm_key_configured': bool(settings.llm_api_key), 'llm_required': settings.llm_required,
            'llm_model': settings.llm_model,
            'database_backend': db_engine.dialect.name,
            'runtime_hostname': socket.gethostname(),
            'note': 'Process health only; use check_setup.py for actual integration checks'}


@app.post('/api/schema/preview', dependencies=[Depends(authenticate)])
async def preview(file: UploadFile = File(...)):
    if not schema_slots.acquire(blocking=False):
        raise HTTPException(429, 'Schema workers busy; retry shortly', headers={'Retry-After': '5'})
    try:
        return await _preview(file)
    finally:
        schema_slots.release()


async def _preview(file):
    content = await read_upload(file)
    def work():
        frame = ingest(content, file.filename or '')
        outcome = HybridProfileLLMMapper().map(frame)
        return {'mapping': outcome.mapping.model_dump(), 'backend': outcome.backend,
                'error': outcome.error, 'profiles': profile(frame)}
    try:
        return await run_in_threadpool(work)
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/analyze', dependencies=[Depends(authenticate)])
async def analyze(file: UploadFile = File(...), mapping_json: str | None = Form(None), horizon: int = Form(28)):
    if not 1 <= horizon <= 366:
        raise HTTPException(422, 'Horizon must be from 1 to 366 days')
    content = await read_upload(file)
    try:
        mapping = SchemaMappingResult.model_validate_json(mapping_json) if mapping_json else None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not job_lock.acquire(blocking=False):
        raise HTTPException(429, 'Forecast worker busy; retry later', headers={'Retry-After': '10'})
    def work():
        captured = []
        result = run_pipeline(content, file.filename or '', forecast_engine, mapping, horizon, on_canonical=captured.append)
        database_start = time.perf_counter()
        with Session.begin() as session:
            run_id = persist_analysis(session, result, captured[0])
        # Includes transaction commit. Returned separately because it is known only after commit.
        return {'run_id': run_id, **result, 'storage_latency_ms': (time.perf_counter() - database_start) * 1000,
                'database_backend': db_engine.dialect.name}
    try:
        return await run_in_threadpool(work)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception('Analysis failed')
        raise HTTPException(503, 'Analysis failed; inspect server logs') from exc
    finally:
        job_lock.release()


@app.post('/api/jobs', status_code=202, dependencies=[Depends(authenticate)])
async def submit_job(request: Request, file: UploadFile = File(...), mapping_json: str | None = Form(None), horizon: int = Form(28)):
    if not 1 <= horizon <= 366:
        raise HTTPException(422, 'Horizon must be from 1 to 366 days')
    content = await read_upload(file)
    try:
        mapping = SchemaMappingResult.model_validate_json(mapping_json) if mapping_json else None
        job_id = await run_in_threadpool(request.app.state.jobs.submit, content, file.filename or '', mapping, horizon)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except QueueFull as exc:
        raise HTTPException(429, str(exc), headers={'Retry-After': '10'}) from exc
    return {'job_id': job_id, 'status': 'queued'}


@app.get('/api/jobs/{job_id}', dependencies=[Depends(authenticate)])
def job_status(job_id: str):
    with Session() as session:
        job = session.get(AnalysisJob, job_id)
        if not job:
            raise HTTPException(404, 'Job not found')
        return {'job_id': job.id, 'status': job.status, 'phase': job.phase, 'run_id': job.run_id,
                'error': job.error, 'created_at': job.created_at, 'updated_at': job.updated_at}


@app.get('/api/runs', dependencies=[Depends(authenticate)])
def list_runs(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    with Session() as session:
        runs = session.scalars(select(AnalysisRun).order_by(AnalysisRun.created_at.desc()).offset(offset).limit(limit))
        return {'total': session.scalar(select(func.count()).select_from(AnalysisRun)),
                'items': [{'run_id': r.id, 'created_at': r.created_at, 'sku_count': len(r.result.get('items', [])),
                           'history_end': r.result.get('history_end'), 'mapping_backend': r.result.get('mapping_backend')}
                          for r in runs]}


@app.get('/api/runs/{run_id}/transactions', dependencies=[Depends(authenticate)])
def run_transactions(run_id: str, part_id: str | None = None, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    with Session() as session:
        if not session.get(AnalysisRun, run_id):
            raise HTTPException(404, 'Run not found')
        query = select(DemandTransaction).where(DemandTransaction.run_id == run_id)
        if part_id:
            query = query.where(DemandTransaction.part_id == part_id)
        rows = session.scalars(query.order_by(DemandTransaction.part_id, DemandTransaction.transaction_date).offset(offset).limit(limit))
        return {'items': [{'part_id': r.part_id, 'transaction_date': r.transaction_date, 'quantity_sold': r.quantity_sold} for r in rows]}


@app.get('/api/runs/{run_id}', dependencies=[Depends(authenticate)])
def get_run(run_id: str):
    with Session() as session:
        run = session.get(AnalysisRun, run_id)
        if run is None:
            raise HTTPException(404, 'Run not found')
        return {'run_id': run.id, **run.result}
