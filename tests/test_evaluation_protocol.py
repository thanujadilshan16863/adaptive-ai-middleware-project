import numpy as np
import pandas as pd
import pytest

from backend.app.services.phase4_forecaster import Forecast, metrics
from backend.benchmarks.docker_stress import memory_mb
from backend.benchmarks.specified_data import workload
from backend.benchmarks.experiments import schema_experiment


def test_weighted_pinball_is_normalized_and_zero_demand_undefined():
    f = Forecast('test', np.array([1., 1.]), np.ones((2, 4)), 'test')
    m = metrics(np.arange(14.), [0., 2.], f)
    assert m['Pinball_Loss_q90'] == pytest.approx(.5)
    assert m['Weighted_Quantile_Loss_q90'] == pytest.approx(1.)
    assert np.isnan(metrics(np.arange(14.), [0., 0.], f)['Weighted_Quantile_Loss_q90'])


def test_docker_memory_units():
    assert memory_mb('1.5GiB') == 1536
    assert memory_mb('1048576B') == 1
    assert memory_mb('100MB') == pytest.approx(100000000/1024**2)
    with pytest.raises(ValueError):
        memory_mb('unknown')


def test_stress_replication_is_dense_and_does_not_change_original():
    grid = pd.DataFrame({'A': [0, 3], 'B': [2, 0]}, index=pd.date_range('2024-01-01', periods=2))
    metadata = pd.DataFrame({'regime':['Smooth','Intermittent'], 'part_name':['a','b'],
        'category':['x','y'], 'current_stock':[5,6], 'lead_time_days':[14,14], 'unit_cost':[2.,3.]}, index=['A','B'])
    frame = workload(grid, metadata, 5)
    assert len(frame) == 10 and frame.part_id.nunique() == 5
    assert not frame.duplicated(['part_id','transaction_date']).any()
    assert list(grid.columns) == ['A','B']


def test_schema_score_uses_only_required_fields_and_actual_file_count(tmp_path):
    pd.DataFrame({'part_id':['A'],'transaction_date':['2024-01-01'],'quantity_sold':[1]}).to_csv(tmp_path/'one.csv',index=False)
    fields = ['part_id','transaction_date','quantity_sold']
    table, raw = schema_experiment(tmp_path, {'mutations':[{'tier':1,'file':'one.csv',
        'truth':dict(zip(fields,fields))}]}, offline=True, fields=fields)
    assert table.n_files.eq(1).all()
    assert table.F1.eq(1).all()
    assert raw.Exact_Match_pct.eq(100).all()
