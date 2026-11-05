"""Validate report completeness and numerical invariants after an experiment run."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def verify(output, require_live=False, require_docker=False):
    output = Path(output)
    tables = output / 'tables' if (output / 'tables').is_dir() else output
    figures = output / 'figures' if (output / 'figures').is_dir() else output
    manifest = json.loads((output / 'run_manifest.json').read_text(encoding='utf-8'))
    table_names = ['table_3_1_dataset_outline', 'table_4_1_schema_benchmark',
                   'table_4_2_forecasting_metrics_by_regime', 'table_4_3_inventory_service_simulation',
                   'table_4_4_system_latency_benchmark']
    for name in table_names:
        assert not pd.read_csv(tables / f'{name}.csv').empty, name
        assert '\\begin{tabular}' in (tables / f'{name}.tex').read_text(encoding='utf-8'), name
    pictures = sorted(figures.glob('fig_4_*.png'))
    assert len(pictures) == 5
    for path in pictures:
        with Image.open(path) as image:
            assert min(image.info['dpi']) >= 299, path
            assert image.width > 1500 and image.height > 1000, path
            image.verify()
    schema = pd.read_csv(tables / 'table_4_1_schema_benchmark.csv')
    assert len(schema) == 9
    assert schema[['Precision', 'Recall', 'F1']].ge(0).all().all()
    assert schema[['Precision', 'Recall', 'F1']].le(1).all().all()
    forecasts = pd.read_csv(tables / 'table_4_2_forecasting_metrics_by_regime.csv')
    regimes = pd.read_csv(output / 'training_regimes.csv').regime.value_counts()
    assert len(forecasts) == len(regimes) * 5
    assert forecasts.n_skus.eq(forecasts.regime.map(regimes)).all()
    assert np.isfinite(forecasts[['MAE', 'RMSE', 'Pinball_Loss_q90']]).all().all()
    inventory = pd.read_csv(tables / 'table_4_3_inventory_service_simulation.csv')
    assert len(inventory) == len(regimes) * 3
    assert inventory.Fill_Rate_pct.between(0, 100).all()
    assert inventory.Average_Stock_Holding.ge(0).all()
    latency = pd.read_csv(tables / 'table_4_4_system_latency_benchmark.csv')
    if manifest.get('latency_backend') == 'docker':
        assert len(latency) == 12 and set(latency.n_skus) == {10, 100, 200}
        assert latency.measurement_backend.eq('docker_http_postgresql').all()
        assert latency.Peak_RAM_MB.gt(0).all()
        audit = json.loads((output / 'docker_stress_audit.json').read_text(encoding='utf-8'))
        assert audit['health']['database_backend'] == 'postgresql'
    else:
        expected_sizes = {s for s in [10, 100, 200] if s <= int(regimes.sum())}
        assert len(latency) == 6 * len(expected_sizes) and set(latency.n_skus) == expected_sizes
    if require_docker:
        assert manifest.get('latency_backend') == 'docker', 'Specification requires actual Docker/PostgreSQL measurements'
    if manifest['mode'] == 'offline':
        assert 'Chronos-Bolt' not in set(forecasts.model)
        assert 'EmpiricalOffline' in set(forecasts.model)
    else:
        assert manifest['mode'] == 'full'
        assert 'Chronos-Bolt' in set(forecasts.model)
        assert 'EmpiricalOffline' not in set(forecasts.model)
        assert not schema.loc[schema.mapper.ne('Fuzzy'), 'backend'].str.startswith('offline_').any()
    if require_live:
        assert manifest['mode'] == 'full', 'Live evidence requires --mode full'
        assert manifest['schema_failed_calls'] == 0, 'Investigate schema API failures before accepting the run'
        raw = pd.read_csv(output / 'schema_raw.csv')
        live = raw[raw.mapper.ne('Fuzzy')]
        assert len(live) == 2 * manifest.get('schema_mutations', 30) and live.attempts.ge(1).all()
        assert live.error.isna().all(), 'Failed provider responses in raw schema results'
        assert live.backend.eq(manifest['llm_model']).all()
    print(f'PASS: five CSV/LaTeX tables, five 300-DPI figures, regimes, metrics and provenance: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', nargs='?', default='outputs')
    parser.add_argument('--require-live', action='store_true', help='Reject offline runs and failed LLM requests')
    parser.add_argument('--require-docker', action='store_true', help='Require measured Docker/PostgreSQL stress results')
    args = parser.parse_args()
    verify(args.output, args.require_live, args.require_docker)
