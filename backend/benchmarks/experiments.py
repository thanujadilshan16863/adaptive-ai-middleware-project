import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from backend.app.pipeline import run_pipeline
from backend.app.schemas import FIELDS
from backend.app.services.phase2_llm_mapper import FuzzyHeuristicMapper, HeaderOnlyLLMMapper, HybridProfileLLMMapper
from backend.app.services.phase3_normalizer import classify
from backend.app.services.phase4_forecaster import Forecast, baselines, metrics
from backend.app.services.phase5_replenish import simulate_inventory, fixed_origin_targets


def schema_experiment(data, manifest, offline=True, fields=None):
    fields = list(fields or FIELDS)
    rows = []
    mappers = {'Fuzzy': FuzzyHeuristicMapper(), 'Header-Only LLM': HeaderOnlyLLMMapper(offline),
               'Hybrid LLM': HybridProfileLLMMapper(offline)}
    for mutation in manifest['mutations']:
        frame = pd.read_csv(Path(data) / mutation['file'])
        for name, mapper in mappers.items():
            start = time.perf_counter()
            result = mapper.map(frame)
            elapsed = (time.perf_counter() - start) * 1000
            predicted = result.mapping.model_dump()
            truth = {k: mutation['truth'][k] for k in fields}
            selected = {k: predicted[k] for k in fields}
            true_positive = sum(selected[k] == truth[k] and truth[k] is not None for k in fields)
            count = sum(v is not None for v in selected.values())
            precision = true_positive / count if count else 0.
            recall = true_positive / sum(v is not None for v in truth.values())
            rows.append({'tier': mutation['tier'], 'mapper': name, 'backend': result.backend,
                         'file': mutation['file'], 'Precision': precision, 'Recall': recall,
                         'F1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
                         'Exact_Match_pct': 100. * (selected == truth), 'Latency_ms': elapsed,
                         'scored_fields': ','.join(fields),
                         'attempts': result.attempts, 'error': result.error,
                         'provider_requests': result.request_count,
                         'predicted_mapping': json.dumps(predicted)})
    raw = pd.DataFrame(rows)
    table = raw.groupby(['tier', 'mapper', 'backend'], sort=False)[
        ['Precision', 'Recall', 'F1', 'Exact_Match_pct', 'Latency_ms']].mean().reset_index()
    counts = raw.groupby(['tier', 'mapper', 'backend'], sort=False).size().rename('n_files').reset_index()
    table = table.merge(counts, on=['tier', 'mapper', 'backend'])
    return table, raw


def forecasting_experiment(grid, metadata, engine):
    split = int(len(grid) * .8)
    test_len = len(grid) - split
    horizon = test_len + int(metadata.lead_time_days.max()) + 1
    predictions = engine.batch([grid[c].values[:split] for c in grid], horizon)
    metric_rows, inventory_rows, examples = [], [], {}
    for sku, foundation in zip(grid, predictions):
        y = grid[sku].values
        train, actual = y[:split], y[split:]
        regime = classify(train)['regime']
        lead = int(metadata.loc[sku, 'lead_time_days'])
        # Identical training-only initialization for all policies; no future stock snapshot.
        initial = float(np.ceil(train.mean() * (lead + 1)))
        all_forecasts = [foundation, *baselines(train, horizon)]
        for f in all_forecasts:
            scored = Forecast(f.model, f.mean[:test_len], f.quantiles[:test_len], f.provenance)
            targets = fixed_origin_targets(f.q(.9), test_len, lead)
            inventory = simulate_inventory(actual, targets, lead, initial)
            metric_rows.append({'part_id': sku, 'regime': regime, 'model': f.model,
                                **metrics(train, actual, scored), 'Fill_Rate_pct': inventory['Fill_Rate_pct'],
                                'provenance': f.provenance})
        sba = next(f for f in all_forecasts if f.model == 'CrostonSBA')
        for policy, values, source in [('Median q50', foundation.q(.5), foundation.model),
                                        ('Quantile q90', foundation.q(.9), foundation.model),
                                        ('SBA point', sba.mean, sba.model)]:
            result = simulate_inventory(actual, fixed_origin_targets(values, test_len, lead), lead, initial)
            inventory_rows.append({'part_id': sku, 'regime': regime, 'policy': policy, 'source_model': source,
                                   **result, 'Average_Stock_Value_LKR': result['Average_Stock_Holding'] * metadata.loc[sku, 'unit_cost']})
        # Choose by training-event count only, before examining holdout performance.
        # This avoids a completely collapsed q10-q90 band in the illustration.
        events = int(np.count_nonzero(train))
        if regime == 'Lumpy' and (not examples or events > examples['training_events']):
            examples = {'sku': sku, 'dates': grid.index, 'actual': y, 'split': split,
                        'foundation': foundation, 'sba': sba, 'training_events': events}
    raw = pd.DataFrame(metric_rows)
    metric_cols = ['MAE', 'RMSE', 'MASE', 'RMSSE', 'Pinball_Loss_q90', 'Fill_Rate_pct']
    table = raw.groupby(['regime', 'model'])[metric_cols].mean().reset_index()
    counts = raw.groupby(['regime', 'model']).agg(n_skus=('part_id', 'size'),
                                                valid_MASE=('MASE', 'count'), valid_RMSSE=('RMSSE', 'count')).reset_index()
    table = table.merge(counts, on=['regime', 'model'])
    sums = raw.groupby(['regime', 'model'])[['Pinball_Sum_q90', 'Absolute_Demand_Units']].sum().reset_index()
    sums['Weighted_Quantile_Loss_q90'] = 2 * sums.Pinball_Sum_q90 / sums.Absolute_Demand_Units.replace(0, np.nan)
    table = table.merge(sums, on=['regime', 'model'])
    inv_raw = pd.DataFrame(inventory_rows)
    inv = inv_raw.groupby(['regime', 'policy', 'source_model']).agg(
        Average_Stock_Holding=('Average_Stock_Holding', 'mean'),
        Average_Stock_Value_LKR=('Average_Stock_Value_LKR', 'mean'),
        Demand_Units=('Demand_Units', 'sum'), Served_Units=('Served_Units', 'sum'),
        n_skus=('part_id', 'size')).reset_index()
    inv['Fill_Rate_pct'] = 100 * inv.Served_Units / inv.Demand_Units.replace(0, np.nan)
    if not examples:
        sku = grid.columns[0]
        y = grid[sku].values
        examples = {'sku': sku, 'dates': grid.index, 'actual': y, 'split': split,
                    'foundation': predictions[0], 'sba': next(f for f in baselines(y[:split], horizon) if f.model == 'CrostonSBA'),
                    'regime': classify(y[:split])['regime'], 'training_events': int(np.count_nonzero(y[:split]))}
    return table, inv, raw, inv_raw, examples


class PhaseProfiler:
    """Sample native process RSS every 2ms, including start/end; not GPU VRAM."""
    def __init__(self):
        self.rows = []

    def __call__(self, name, function):
        process = psutil.Process()
        initial = process.memory_info().rss
        samples = [initial]
        done = threading.Event()
        def sample():
            while not done.wait(.002):
                samples.append(process.memory_info().rss)
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        start = time.perf_counter()
        try:
            return function()
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            samples.append(process.memory_info().rss)
            done.set()
            thread.join()
            self.rows.append({'phase': name, 'Latency_ms': elapsed, 'Peak_RAM_MB': max(samples) / 1024**2,
                              'Incremental_RAM_MB': (max(samples) - initial) / 1024**2})


def latency_experiment(transactions, metadata, engine, repeats=3, offline=True):
    rows = []
    # Warm-up is explicit and excluded; cold model-load time is captured separately by caller.
    engine.forecast(np.ones(365), 29)
    ordered = []
    groups = [list(g.part_id) for _, g in metadata.reset_index().groupby('regime')]
    for i in range(max(map(len, groups))):
        ordered.extend(group[i] for group in groups if i < len(group))
    for size in [10, 100, 200]:
        if size > len(ordered):
            continue
        content = transactions[transactions.part_id.isin(ordered[:size])].to_csv(index=False).encode()
        for repetition in range(repeats):
            profiler = PhaseProfiler()
            result = run_pipeline(content, 'latency.csv', engine, observer=profiler, offline_llm=offline)
            rows.extend(dict(row, n_skus=size, repetition=repetition + 1,
                             model=result['items'][0]['model'], mapping_backend=result['mapping_backend']) for row in profiler.rows)
    raw = pd.DataFrame(rows)
    table = raw.groupby(['n_skus', 'phase', 'model', 'mapping_backend']).agg(
        Latency_ms=('Latency_ms', 'median'), Latency_p95_ms=('Latency_ms', lambda s: s.quantile(.95)),
        Peak_RAM_MB=('Peak_RAM_MB', 'max'), Incremental_RAM_MB=('Incremental_RAM_MB', 'max'),
        repeats=('repetition', 'count')).reset_index()
    return table, raw
