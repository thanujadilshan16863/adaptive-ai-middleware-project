import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.schemas import FIELDS, SchemaMappingResult
from backend.app.services.phase1_ingestion import ingest
from backend.app.services.phase2_llm_mapper import HybridProfileLLMMapper, HeaderOnlyLLMMapper
from backend.app.services.phase3_normalizer import classify, canonicalize, zero_fill
from backend.app.services.phase4_forecaster import Forecast, ForecastEngine, baseline_point, metrics
from backend.app.services.phase5_replenish import simulate_inventory, fixed_origin_targets, replenish
from backend.benchmarks.generate_synthetic_data import generate


def test_classification_and_boundaries():
    result = classify([0, 2, 0, 4])
    assert result['ADI'] == 2
    assert result['CV2'] == pytest.approx(2 / 9)
    assert result['regime'] == 'Intermittent'
    assert classify([1] * 100)['regime'] == 'Smooth'
    assert classify([0] * 10)['regime'] == 'All-zero'
    assert classify([0, 1])['regime'] == 'Insufficient'
    assert classify([1] * 100 + [0] * 32)['regime'] == 'Intermittent'


def test_zero_fill_global_window_and_duplicates():
    df = pd.DataFrame({'part_id': ['a', 'a', 'b'], 'transaction_date': pd.to_datetime(['2023-01-02', '2023-01-02', '2023-01-04']), 'quantity_sold': [2, 3, 7]})
    grid = zero_fill(df, '2023-01-01', '2023-01-05')
    assert grid.a.tolist() == [0, 5, 0, 0, 0]
    assert grid.b.tolist() == [0, 0, 0, 7, 0]


def test_baselines_by_hand():
    y = np.array([0., 2., 0., 4.])
    assert baseline_point(y, 2, 'CrostonClassic')[0] == pytest.approx(1.1)
    assert baseline_point(y, 2, 'CrostonSBA')[0] == pytest.approx(1.045)
    assert baseline_point(y, 2, 'TSB')[0] == pytest.approx(2.2 * .505)
    assert baseline_point(np.zeros(12), 2, 'TSB').tolist() == [0, 0]
    assert baseline_point(np.arange(1, 9), 3, 'SeasonalNaive').tolist() == [2, 3, 4]


def test_metrics_and_undefined_scales():
    f = Forecast('test', np.array([2., 2.]), np.array([[1., 2., 3., 4.]] * 2), 'test')
    m = metrics(np.arange(1., 11.), np.array([0., 4.]), f, season_length=1)
    assert m['MAE'] == 2 and m['RMSE'] == 2
    assert m['MASE'] == 2 and m['RMSSE'] == 2
    assert m['Pinball_Loss_q90'] == pytest.approx(.2)
    assert np.isnan(metrics(np.ones(10), np.array([0., 4.]), f)['MASE'])


def test_inventory_pipeline_and_conservation():
    result = simulate_inventory([3, 3, 3], [6, 6, 6], lead_time_days=1, initial_stock=3)
    assert result['Fill_Rate_pct'] == 100
    assert result['Average_Stock_Holding'] == 2
    shortage = simulate_inventory([5, 5, 5], [10, 10, 10], 2, 0)
    assert shortage['Served_Units'] == 5
    assert shortage['Served_Units'] + shortage['Lost_Units'] == shortage['Demand_Units']
    assert np.isnan(simulate_inventory([0], [0], 1, 0)['Fill_Rate_pct'])
    assert fixed_origin_targets(np.ones(6), 3, 2).tolist() == [3, 3, 3]
    assert fixed_origin_targets(np.arange(6), 3, 2).tolist() == [6, 9, 12]


def test_mapping_validation_and_fallback():
    df = pd.DataFrame({'COL_A': ['TOY-123', 'HON-234'], 'COL_B': ['2023-01-01', '2023-01-02'], 'COL_C': [2, 3]})
    h = HybridProfileLLMMapper(offline=True).map(df)
    assert h.mapping.part_id == 'COL_A'
    assert h.mapping.transaction_date == 'COL_B'
    assert h.mapping.quantity_sold is None
    assert HeaderOnlyLLMMapper(offline=True).map(df).mapping.part_id is None
    with pytest.raises(ValueError):
        SchemaMappingResult(**{k: 'same' for k in FIELDS})


def test_ingestion_rejects_duplicate_headers():
    with pytest.raises(ValueError, match='Duplicate'):
        ingest(b'a,a,c\n1,2,3\n', 'x.csv')


def test_generator_reproducible_and_regimes(tmp_path):
    manifest = generate(tmp_path, n_skus=8)
    original = (tmp_path / 'raw/automotive_transactions.csv').read_bytes()
    generate(tmp_path, n_skus=8)
    assert original == (tmp_path / 'raw/automotive_transactions.csv').read_bytes()
    assert len(manifest['mutations']) == 30
    dense = pd.read_csv(tmp_path / 'processed/daily_demand.csv', index_col=0)
    assert dense.shape == (730, 8)
    regimes = [classify(dense[c].values[:584])['regime'] for c in dense]
    assert set(regimes) == {'Smooth', 'Erratic', 'Intermittent', 'Lumpy'}
    for item in manifest['mutations']:
        columns = pd.read_csv(tmp_path / item['file'], nrows=0).columns
        assert set(item['truth'].values()) <= set(columns)


def test_api_roundtrip_and_validation(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    import backend.app.main as api
    db = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    monkeypatch.setattr(api, 'db_engine', db)
    monkeypatch.setattr(api, 'Session', sessionmaker(db))
    monkeypatch.setattr(api, 'forecast_engine', ForecastEngine('offline'))
    monkeypatch.setattr(api.settings, 'api_key', 'test-key')
    generate(tmp_path, n_skus=4)
    content = (tmp_path / 'raw/automotive_transactions.csv').read_bytes()
    with TestClient(api.app) as client:
        assert client.post('/api/analyze', files={'file': ('data.csv', content)}).status_code == 401
        response = client.post('/api/analyze', files={'file': ('data.csv', content)}, headers={'X-API-Key': 'test-key'})
        assert response.status_code == 200, response.text
        result = response.json()
        assert len(result['items']) == 4
        assert result['items'][0]['model'] == 'EmpiricalOffline'
        fetched = client.get('/api/runs/' + result['run_id'], headers={'X-API-Key': 'test-key'})
        assert result['storage_latency_ms'] >= 0
        assert result['database_backend'] == 'sqlite'
        # Commit timing is measured after persistence and is response telemetry, not the snapshot.
        assert fetched.json() == {k: v for k, v in result.items() if k not in ('storage_latency_ms', 'database_backend')}
        assert client.post('/api/analyze', files={'file': ('x.exe', b'123')}, headers={'X-API-Key': 'test-key'}).status_code == 422
    db.dispose()


def test_llm_correction_once(monkeypatch):
    import backend.app.services.phase2_llm_mapper as module
    monkeypatch.setattr(module.settings, 'llm_api_key', 'test')
    calls = []
    valid = dict.fromkeys(FIELDS)
    valid.update(part_id='sku', transaction_date='date', quantity_sold='qty')
    def handler(request):
        calls.append(json.loads(request.content))
        value = dict(valid, part_id='invented') if len(calls) == 1 else valid
        import httpx
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(value)}}]})
    import httpx
    original = httpx.Client
    monkeypatch.setattr(module.httpx, 'Client', lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    result = module.HybridProfileLLMMapper().map(pd.DataFrame({'sku': ['A'], 'date': ['2023-01-01'], 'qty': [2]}))
    assert result.attempts == 2 and result.mapping.part_id == 'sku'
    assert 'profiles' in json.loads(calls[0]['messages'][1]['content'])
    assert len(calls) == 2
