"""Run real forecasting and adaptive ablation independently of an LLM account."""
import argparse
import json
from pathlib import Path
import time
import importlib.metadata
import pandas as pd

from backend.app.config import settings
from backend.app.services.phase4_forecaster import ForecastEngine
from backend.benchmarks.generate_synthetic_data import generate
from backend.benchmarks.experiments import forecasting_experiment
from backend.benchmarks.adaptive_eval import adaptive_experiment
from backend.benchmarks.artifacts import export_table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['offline','chronos'],default='chronos')
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path,default=Path('outputs/chronos_validation'))
    args=parser.parse_args()
    if not (args.data/'processed/daily_demand.csv').exists():
        generate(args.data)
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    grid=pd.read_csv(args.data/'processed/daily_demand.csv',index_col=0,parse_dates=True)
    metadata=pd.read_csv(args.data/'raw/sku_metadata.csv').set_index('part_id')
    engine=ForecastEngine(args.mode)
    print('Running fixed-origin model comparison...',flush=True)
    summary,inventory,raw,inventory_raw,_=forecasting_experiment(grid,metadata,engine)
    export_table(summary,'table_4_2_forecasting_metrics_by_regime',args.output)
    export_table(inventory,'table_4_3_inventory_service_simulation',args.output)
    raw.to_csv(args.output/'forecasting_raw.csv',index=False)
    inventory_raw.to_csv(args.output/'inventory_raw.csv',index=False)
    print('Running nested adaptive-model ablation...',flush=True)
    summary,raw,paired,selection=adaptive_experiment(grid,engine)
    export_table(summary,'table_4_5_adaptive_selection',args.output)
    export_table(paired,'table_4_6_paired_bootstrap',args.output)
    raw.to_csv(args.output/'adaptive_raw.csv',index=False)
    (args.output/'selection_audit.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    manifest={'scope':'forecasting and adaptive ablation only; no LLM results',
              'mode':args.mode,'model':settings.chronos_model if args.mode=='chronos' else 'EmpiricalOffline',
              'strategy':'inner temporal validation, refit, untouched outer 20% holdout',
              'skus':len(grid.columns),'days':len(grid),'elapsed_seconds':time.perf_counter()-start,
              'versions':{name:importlib.metadata.version(name) for name in ['numpy','pandas','chronos-forecasting','torch'] if importlib.util.find_spec(name.replace('-forecasting',''))}}
    (args.output/'forecast_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Completed {len(grid.columns)} SKUs in {manifest["elapsed_seconds"]:.1f}s: {args.output}',flush=True)


if __name__=='__main__':
    main()
