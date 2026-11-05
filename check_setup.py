"""Local configuration checks plus explicit opt-in integration smoke tests.

Default: no provider calls, downloads or database connection. --llm sends only
built-in synthetic sample values. --chronos may download model weights.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from backend.app.config import settings
from backend.app.services.phase2_llm_mapper import HybridProfileLLMMapper
from backend.app.services.phase4_forecaster import ForecastEngine


def check_llm():
    if not settings.llm_api_key:
        raise ValueError('Set LLM_API_KEY in the project-root .env first')
    frame = pd.DataFrame({
        'OEM_Part_No': ['TOY-10001', 'HON-10002', 'NIS-10003'],
        'Sale_Dt': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Qty_Sold': [2, 1, 4], 'Description': ['Oil Filter', 'Brake Pad', 'Water Pump'],
        'Part_Group': ['Filters', 'Braking', 'Cooling'], 'Stock_On_Hand': [40, 25, 10],
        'Lead_Days': [7, 14, 21], 'Unit_Price': [1500.5, 4200.75, 12500.25]})
    outcome = HybridProfileLLMMapper().map(frame)
    if outcome.error:
        raise RuntimeError(outcome.error)
    expected = dict(zip(['part_id', 'transaction_date', 'quantity_sold', 'part_name',
                         'category', 'current_stock', 'lead_time_days', 'unit_cost'], frame.columns))
    if outcome.mapping.model_dump() != expected:
        raise RuntimeError('Provider responded, but did not correctly map the simple smoke-test schema')
    if outcome.attempts < 1 or outcome.backend.startswith('offline_'):
        raise RuntimeError('A live LLM response was required; fallback is not acceptable')
    return {'model': outcome.backend, 'attempts': outcome.attempts, 'mapping': outcome.mapping.model_dump()}


def check_chronos():
    import torch
    if settings.chronos_device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected but unavailable; use CHRONOS_DEVICE=cpu or install compatible GPU PyTorch')
    rng = np.random.default_rng(42)
    histories = [rng.poisson(8, 584).astype(float),
                 (rng.random(584) < .12) * np.ceil(rng.lognormal(1, 1.2, 584))]
    engine = ForecastEngine('chronos')
    shapes = {}
    for horizon in (28, 175):
        forecasts = engine.batch(histories, horizon)
        if len(forecasts) != len(histories):
            raise RuntimeError('Incorrect Chronos batch length')
        for forecast in forecasts:
            if forecast.model != 'Chronos-Bolt' or forecast.quantiles.shape != (horizon, 4):
                raise RuntimeError('Incorrect Chronos output contract')
            if not np.isfinite(forecast.quantiles).all() or not np.isfinite(forecast.mean).all():
                raise RuntimeError('Nonfinite Chronos forecast')
            if (forecast.quantiles < 0).any() or (np.diff(forecast.quantiles, axis=1) < 0).any():
                raise RuntimeError('Negative or crossing quantiles')
        shapes[str(horizon)] = [len(forecasts), horizon, 4]
    return {'model': settings.chronos_model, 'device': settings.chronos_device,
            'torch_version': torch.__version__, 'tested_shapes': shapes,
            'note': 'Real inference smoke test, not an accuracy benchmark'}


def check_database():
    from sqlalchemy import create_engine, text
    engine = create_engine(settings.database_url)
    try:
        with engine.connect() as connection:
            value = connection.execute(text('SELECT 1')).scalar_one()
        if value != 1:
            raise RuntimeError('Database query failed')
        return {'dialect': engine.dialect.name, 'query': 'SELECT 1', 'result': value}
    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm', action='store_true', help='Make a potentially billable provider call using synthetic samples')
    parser.add_argument('--chronos', action='store_true', help='Load/download Chronos and test 28/175-day forecasts')
    parser.add_argument('--database', action='store_true', help='Connect to the configured database and run SELECT 1')
    parser.add_argument('--output', type=Path, help='Save the checks as JSON; no API keys are included')
    args = parser.parse_args()
    report = {'python': platform.python_version(), 'executable': sys.executable,
              'env_file_exists': Path('.env').is_file(), 'llm_key_configured': bool(settings.llm_api_key),
              'llm_model': settings.llm_model, 'llm_json_schema': settings.llm_json_schema,
              'llm_required': settings.llm_required, 'forecast_mode': settings.forecast_mode,
              'torch_installed': importlib.util.find_spec('torch') is not None,
              'chronos_installed': importlib.util.find_spec('chronos') is not None,
              'middleware_key_configured': bool(settings.api_key), 'checks': {}}
    failed = False
    for name, enabled, function in [('llm', args.llm, check_llm), ('chronos', args.chronos, check_chronos),
                                     ('database', args.database, check_database)]:
        if not enabled:
            report['checks'][name] = {'status': 'not_run'}
            continue
        print(f'Checking {name}...', flush=True)
        start = time.perf_counter()
        try:
            result = function()
            report['checks'][name] = {'status': 'passed', 'seconds': time.perf_counter() - start, **result}
        except Exception as exc:
            # Do not echo connection URLs, provider response bodies or credentials.
            failed = True
            report['checks'][name] = {'status': 'failed', 'error_type': type(exc).__name__,
                'hint': {'llm': 'Check key, model access, endpoint, JSON mode, quota and network; see SETUP_AND_RUN_GUIDE.md.',
                         'chronos': 'Check optional dependencies, cache permissions, network and device; see SETUP_AND_RUN_GUIDE.md.',
                         'database': 'Check database server, credentials and connection URL; see SETUP_AND_RUN_GUIDE.md.'}[name]}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
