import json
import threading
import time
from datetime import datetime, timezone

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select, func
from sqlalchemy.orm import sessionmaker

from backend.app.database import AnalysisRun, AnalysisJob, DemandTransaction, PartSnapshot, migrate
from backend.app.jobs import JobManager, QueueFull
from backend.app.services.phase4_forecaster import AdaptiveForecastEngine, ForecastEngine
from backend.benchmarks.generate_synthetic_data import generate


def test_migration_preserves_legacy_runs(tmp_path):
    engine = create_engine('sqlite:///' + str(tmp_path / 'old.db'))
    AnalysisRun.__table__.create(engine)
    sessions = sessionmaker(engine)
    with sessions.begin() as session:
        session.add(AnalysisRun(id='legacy', result={'items': []}))
    migrate(engine)
    migrate(engine)
    assert {'analysis_runs','analysis_jobs','part_snapshots','demand_transactions','alembic_version'} <= set(inspect(engine).get_table_names())
    with sessions() as session:
        assert session.get(AnalysisRun,'legacy').result == {'items': []}
    engine.dispose()


def test_adaptive_validation_never_includes_validation_targets():
    class RecordingEngine(ForecastEngine):
        def __init__(self):
            super().__init__('offline')
            self.received = []
        def batch(self, histories, horizon=28):
            self.received.append(([np.array(y).copy() for y in histories], horizon))
            return super().batch(histories,horizon)
    foundation = RecordingEngine()
    y = np.tile([0.,0.,2.,0.,0.,4.,0.], 20)
    result = AdaptiveForecastEngine(foundation).forecast(y, 14)
    sent,horizon = foundation.received[0]
    assert horizon == 28
    np.testing.assert_array_equal(sent[0], y[:-28])
    assert result.selection['selected_model'] == result.model
    assert result.selection['selection_score'] == min(result.selection['candidate_scores'].values())
    assert result.quantiles.shape == (14,4)
    assert np.isfinite(result.mean).all()


def test_jobs_persist_canonical_and_expose_history(tmp_path, monkeypatch):
    import backend.app.main as api
    engine = create_engine('sqlite:///' + str(tmp_path / 'jobs.db'), connect_args={'check_same_thread':False})
    monkeypatch.setattr(api,'db_engine',engine)
    monkeypatch.setattr(api,'Session',sessionmaker(engine))
    monkeypatch.setattr(api,'forecast_engine',AdaptiveForecastEngine(ForecastEngine('offline')))
    monkeypatch.setattr(api.settings,'job_directory',str(tmp_path/'uploads'))
    monkeypatch.setattr(api.settings,'api_key','test-key')
    generate(tmp_path/'data',n_skus=4)
    content=(tmp_path/'data/raw/automotive_transactions.csv').read_bytes()
    headers={'X-API-Key':'test-key'}
    with TestClient(api.app) as client:
        assert client.get('/api/runs').status_code==401
        submitted=client.post('/api/jobs',files={'file':('data.csv',content)},headers=headers)
        assert submitted.status_code==202,submitted.text
        job_id=submitted.json()['job_id']
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            job=client.get('/api/jobs/'+job_id,headers=headers).json()
            if job['status'] in ('completed','failed'):
                break
            time.sleep(.05)
        assert job['status']=='completed',job
        saved=client.get('/api/runs/'+job['run_id'],headers=headers)
        assert saved.status_code==200
        assert all(item['selection'] for item in saved.json()['items'])
        assert client.get('/api/runs',headers=headers).json()['total']==1
        rows=client.get('/api/runs/'+job['run_id']+'/transactions',headers=headers).json()['items']
        assert len(rows)==100
        with api.Session() as session:
            assert session.scalar(select(func.count()).select_from(PartSnapshot))==4
            assert session.scalar(select(func.count()).select_from(DemandTransaction))>100
        assert not list((tmp_path/'uploads').glob('*.upload'))
    engine.dispose()


def test_queue_recovery_and_capacity(tmp_path):
    engine=create_engine('sqlite:///'+str(tmp_path/'q.db'),connect_args={'check_same_thread':False})
    migrate(engine)
    sessions=sessionmaker(engine)
    with sessions.begin() as session:
        session.add(AnalysisJob(id='00000000-0000-0000-0000-000000000001',status='running',phase='4_forecasting'))
    lock=threading.Lock()
    manager=JobManager(sessions,ForecastEngine('offline'),lock,tmp_path/'q',1)
    with sessions() as session:
        assert session.get(AnalysisJob,'00000000-0000-0000-0000-000000000001').status=='failed'
    lock.acquire()
    try:
        manager.submit(b'bad','file.exe',None,28)
        manager.submit(b'bad','file.exe',None,28)
        with pytest.raises(QueueFull):
            manager.submit(b'bad','file.exe',None,28)
    finally:
        lock.release()
        manager.close()
    assert not list((tmp_path/'q').glob('*.upload'))
    engine.dispose()


def test_llm_transport_retry_separate_from_correction(monkeypatch):
    import backend.app.services.phase2_llm_mapper as mapper
    from backend.app.schemas import FIELDS
    import pandas as pd
    monkeypatch.setattr(mapper.settings,'llm_api_key','test')
    monkeypatch.setattr(mapper.time,'sleep',lambda delay:None)
    calls=[]
    def handler(request):
        calls.append(request)
        if len(calls)==1:
            return httpx.Response(429,headers={'Retry-After':'1'})
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(dict.fromkeys(FIELDS))}}]})
    original=httpx.Client
    monkeypatch.setattr(mapper.httpx,'Client',lambda **kw: original(transport=httpx.MockTransport(handler),**kw))
    outcome=mapper.HeaderOnlyLLMMapper().map(pd.DataFrame({'sku':['A']}))
    assert outcome.attempts==1 and outcome.request_count==2 and outcome.error is None
