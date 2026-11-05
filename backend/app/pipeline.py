import json
import time
import numpy as np

from backend.app.services.phase1_ingestion import ingest
from backend.app.services.phase2_llm_mapper import HybridProfileLLMMapper
from backend.app.services.phase3_normalizer import canonicalize, zero_fill, classify
from backend.app.services.phase5_replenish import replenish


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def run_pipeline(content, filename, engine, mapping=None, horizon=28, observer=None, offline_llm=False,
                 on_canonical=None, progress=None):
    timings = {}

    def phase(name, fn):
        if progress:
            progress(name)
        start = time.perf_counter()
        value = observer(name, fn) if observer else fn()
        timings[name] = (time.perf_counter() - start) * 1000
        return value

    frame = phase('1_ingestion', lambda: ingest(content, filename))
    outcome = phase('2_schema_alignment', lambda: HybridProfileLLMMapper(offline=offline_llm).map(frame)) if mapping is None else None
    mapping = mapping if mapping is not None else outcome.mapping

    def normalize():
        canonical = canonicalize(frame, mapping)
        return canonical, zero_fill(canonical)

    canonical, grid = phase('3_normalization', normalize)
    if on_canonical:
        on_canonical(canonical)
    metadata = canonical.groupby('part_id', sort=True).tail(1).set_index('part_id')
    required_horizon = max(horizon, int(metadata.lead_time_days.max()) + 1)
    forecasts = phase('4_forecasting', lambda: engine.batch([grid[c].values for c in grid], required_horizon))

    def decisions():
        return [dict(part_id=sku, part_name=str(metadata.loc[sku, 'part_name']),
                     category=str(metadata.loc[sku, 'category']), **classify(grid[sku]),
                     model=f.model, provenance=f.provenance, mean_method=f.mean_method,
                     selection=f.selection, current_stock=float(metadata.loc[sku, 'current_stock']),
                     lead_time_days=int(metadata.loc[sku, 'lead_time_days']),
                     history={'dates': grid.index[-90:].strftime('%Y-%m-%d').tolist(),
                              'actual': grid[sku].values[-90:]},
                     forecast={'mean': f.mean, 'q10': f.q(.1), 'q50': f.q(.5), 'q80': f.q(.8), 'q90': f.q(.9)},
                     **replenish(f, float(metadata.loc[sku, 'current_stock']), int(metadata.loc[sku, 'lead_time_days'])))
                for sku, f in zip(grid.columns, forecasts)]

    items = phase('5_replenishment', decisions)
    result = phase('6_json_delivery', lambda: json.loads(json.dumps(json_safe({
        'mapping': mapping.model_dump(), 'mapping_backend': outcome.backend if outcome else 'user_reviewed',
        'frequency': 'D', 'history_start': str(grid.index[0].date()), 'history_end': str(grid.index[-1].date()),
        'forecast_start': str((grid.index[-1] + __import__('pandas').Timedelta(days=1)).date()),
        'horizon': required_horizon, 'items': items}), allow_nan=False)))
    result['phase_latency_ms'] = timings
    return result
