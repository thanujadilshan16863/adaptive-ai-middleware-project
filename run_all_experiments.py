"""One command generates all five report tables, LaTeX and five 300-DPI figures."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import time

import pandas as pd

from backend.app.config import settings
from backend.app.schemas import FIELDS, REQUIRED
from backend.app.services.phase3_normalizer import classify
from backend.app.services.phase4_forecaster import ForecastEngine, baselines
from backend.benchmarks.generate_synthetic_data import generate
from backend.benchmarks.experiments import schema_experiment, forecasting_experiment, latency_experiment
from backend.benchmarks.artifacts import export_table, figures
from backend.benchmarks.adaptive_eval import adaptive_experiment
from backend.benchmarks.specified_data import load_specified_data
from backend.benchmarks.docker_stress import docker_latency_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['offline', 'full'], default='offline')
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--dataset-dir', type=Path, help='Folder containing the four supplied specification CSVs')
    parser.add_argument('--latency-backend', choices=['process', 'docker'], default=None)
    parser.add_argument('--api-url', default='http://127.0.0.1:8001')
    parser.add_argument('--synthetic-lumpy-illustration', action='store_true', help='Use a labelled supplemental lumpy SKU from --data for figures only')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.mode == 'full' and not settings.llm_api_key:
        parser.error('Full mode requires LLM_API_KEY; offline results are not LLM/Chronos evidence')
    args.output.mkdir(parents=True, exist_ok=True)
    table_dir, figure_dir = args.output / 'tables', args.output / 'figures'
    table_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    latency_backend = args.latency_backend or ('docker' if args.dataset_dir else 'process')
    start = time.perf_counter()
    engine = ForecastEngine('chronos' if args.mode == 'full' else 'offline')
    load_start = time.perf_counter()
    if args.mode == 'full':
        print('Loading Chronos before making billable schema requests...', flush=True)
        engine.load()
    model_load_ms = (time.perf_counter() - load_start) * 1000
    if args.dataset_dir:
        print('Reading supplied evaluation files without modifying them...', flush=True)
        manifest, grid, metadata, transactions, dataset_provenance = load_specified_data(args.dataset_dir)
        schema_dir = args.dataset_dir
    else:
        print('Generating 200 SKUs, 730 days and 30 schema mutations...', flush=True)
        manifest = generate(args.data, args.seed)
        grid = pd.read_csv(args.data / 'processed/daily_demand.csv', index_col=0, parse_dates=True)
        metadata = pd.read_csv(args.data / 'raw/sku_metadata.csv').set_index('part_id')
        transactions = pd.read_csv(args.data / 'raw/automotive_transactions.csv')
        schema_dir = args.data
        dataset_provenance = {'source': 'generated synthetic', 'days': len(grid), 'n_skus': grid.shape[1]}
    regimes = pd.DataFrame([{'part_id': c, **classify(grid[c].values[:int(.8 * len(grid))])} for c in grid])
    regime_counts = regimes.regime.value_counts().reindex(['Smooth', 'Erratic', 'Intermittent', 'Lumpy'], fill_value=0)
    outline = pd.DataFrame([{'section': 'variable', 'item': c, 'value': str(transactions[c].dtype),
                             'description': {'quantity_sold': 'Nonnegative daily units; missing dates mean zero demand',
                                             'current_stock': 'Synthetic opening stock scenario, not historical stock observations',
                                             'unit_cost': 'Synthetic unit acquisition cost in LKR'}.get(c, c.replace('_', ' '))} for c in FIELDS]
        + [{'section': 'training_regime', 'item': k, 'value': int(v), 'description': 'SKU count; sample variance ddof=1'}
           for k, v in regime_counts.items()]
        + [{'section': 'dataset', 'item': 'calendar_days', 'value': 730, 'description': '104 complete weeks plus 2 days'},
           {'section': 'dataset', 'item': 'SKUs', 'value': grid.shape[1], 'description': 'Observed unique SKUs'},
           {'section': 'dataset', 'item': 'transaction_rows', 'value': len(transactions), 'description': 'Positive-demand observations'},
           {'section': 'dataset', 'item': 'dense_cells', 'value': grid.size, 'description': 'Zero-filled SKU-day grid'}])
    outline.loc[outline.item.eq('calendar_days'), ['value', 'description']] = [len(grid), 'Observed daily grid including leap days']
    if args.dataset_dir:
        outline.loc[outline.item.eq('unit_cost'), 'description'] = 'unit_price proxy; acquisition cost/currency unverified'
        outline.loc[outline.item.eq('lead_time_days'), 'description'] = 'Assumed 14-day lead-time scenario; not measured'
    export_table(outline, 'table_3_1_dataset_outline', table_dir)
    print('Running schema ablation...', flush=True)
    schema, schema_raw = schema_experiment(schema_dir, manifest, args.mode == 'offline', fields=REQUIRED)
    (args.output / 'schema_ground_truth.json').write_text(json.dumps(manifest['mutations'], indent=2), encoding='utf-8')
    export_table(schema, 'table_4_1_schema_benchmark', table_dir)
    schema_raw.to_csv(args.output / 'schema_raw.csv', index=False)
    print('Running temporal holdout forecasts and inventory simulations...', flush=True)
    forecast, inventory, forecast_raw, inventory_raw, example = forecasting_experiment(grid, metadata, engine)
    illustration_provenance = {'source': 'evaluation dataset', 'sku': example['sku']}
    if args.synthetic_lumpy_illustration:
        supplemental_path = args.data / 'processed/daily_demand.csv'
        supplemental = pd.read_csv(supplemental_path, index_col=0, parse_dates=True)
        split = int(.8 * len(supplemental))
        candidates = [c for c in supplemental if classify(supplemental[c].values[:split])['regime'] == 'Lumpy']
        if not candidates:
            raise ValueError('Supplemental dataset contains no lumpy training SKU')
        sku = max(candidates, key=lambda c: int(supplemental[c].values[:split].astype(bool).sum()))
        y = supplemental[sku].values
        horizon = len(y) - split
        example = {'sku': sku, 'dates': supplemental.index, 'actual': y, 'split': split,
                   'foundation': engine.forecast(y[:split], horizon),
                   'sba': next(f for f in baselines(y[:split], horizon) if f.model == 'CrostonSBA'),
                   'regime': 'Lumpy', 'supplemental': True}
        illustration_provenance = {'source': 'supplemental synthetic illustration only; excluded from tables',
            'sku': sku, 'sha256': hashlib.sha256(supplemental_path.read_bytes()).hexdigest(),
            'classification': classify(y[:split])}
    if args.dataset_dir:
        inventory.rename(columns={'Average_Stock_Value_LKR': 'Average_Stock_Value_Proxy'}, inplace=True)
        inventory_raw.rename(columns={'Average_Stock_Value_LKR': 'Average_Stock_Value_Proxy'}, inplace=True)
    export_table(forecast, 'table_4_2_forecasting_metrics_by_regime', table_dir)
    export_table(inventory, 'table_4_3_inventory_service_simulation', table_dir)
    forecast_raw.to_csv(args.output / 'forecasting_raw.csv', index=False)
    inventory_raw.to_csv(args.output / 'inventory_raw.csv', index=False)
    regimes.to_csv(args.output / 'training_regimes.csv', index=False)
    print(f'Profiling middleware using {latency_backend} measurements...', flush=True)
    if latency_backend == 'docker':
        latency, latency_raw, stress = docker_latency_experiment(grid, metadata, args.api_url, args.repeats, args.mode == 'full')
        (args.output / 'docker_stress_audit.json').write_text(json.dumps(stress, indent=2), encoding='utf-8')
    else:
        latency, latency_raw = latency_experiment(transactions, metadata, engine, args.repeats, args.mode == 'offline')
        latency['measurement_backend'] = 'in_process_no_database'
    export_table(latency, 'table_4_4_system_latency_benchmark', table_dir)
    latency_raw.to_csv(args.output / 'latency_raw.csv', index=False)
    figures(figure_dir, grid, regimes, schema, latency, example, args.mode == 'offline')
    print('Evaluating adaptive selection on the untouched outer holdout...', flush=True)
    adaptive_summary, adaptive_raw, paired, selection = adaptive_experiment(grid, engine, args.seed)
    export_table(adaptive_summary, 'table_4_5_adaptive_selection', table_dir)
    export_table(paired, 'table_4_6_paired_bootstrap', table_dir)
    adaptive_raw.to_csv(args.output / 'adaptive_raw.csv', index=False)
    (args.output / 'selection_audit.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    evidence = {'mode': args.mode, 'seed': args.seed, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages,
                'model': settings.chronos_model if args.mode == 'full' else 'EmpiricalOffline',
                'llm_model': settings.llm_model if args.mode == 'full' else 'offline rules',
                'model_load_ms': model_load_ms, 'total_seconds': time.perf_counter() - start,
                'data_sha256': hashlib.sha256(transactions.to_csv(index=False).encode()).hexdigest(),
                'dataset': dataset_provenance, 'schema_mutations': len(manifest['mutations']),
                'illustration': illustration_provenance,
                'schema_scored_fields': REQUIRED, 'latency_backend': latency_backend,
                'schema_failed_calls': int(schema_raw.error.notna().sum()),
                'protocol': {'split': f'{int(.8 * len(grid))} training days / {len(grid)-int(.8 * len(grid))} test days; fixed origin', 'scale_lag': 7,
                             'weighted_quantile_loss': '2 * sum(q90 pinball) / sum(abs(actual)); pooled within regime; undefined when denominator zero',
                             'variance_ddof': 1, 'inventory': 'daily lost sales, fixed-origin targets, L+1 protection period',
                             'quantiles': 'marginal; summed upper quantiles assume comonotonic demand',
                             'latency': 'See docker_stress_audit.json for Docker HTTP measurements; process mode excludes network/database'},
                'limitations': ['Single supplied/generated dataset; no claim of external validity',
                                'Offline substitutes are not evidence of LLM or Chronos performance',
                                'Baseline quantiles use training one-step residual calibration, not native distributions',
                                'Daily horizon exceeds Bolt native 64 steps; extension is model-recursive',
                                'Three latency repetitions are descriptive, not robust statistical inference']}
    (args.output / 'run_manifest.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(f'Complete: {args.output.resolve()} ({evidence["total_seconds"]:.1f}s); mode={args.mode}', flush=True)


if __name__ == '__main__':
    main()
