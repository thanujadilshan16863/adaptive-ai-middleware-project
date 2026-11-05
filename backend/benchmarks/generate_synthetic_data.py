"""Seeded automotive data; labels are verified on the training window only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.schemas import FIELDS
from backend.app.services.phase3_normalizer import classify

REGIMES = ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']
PARTS = [('Oil Filter', 'Brake Pad'), ('Fleet Spark Plug', 'Coolant'),
         ('Water Pump', 'Alternator'), ('Collision Body Panel', 'Overhaul Kit')]
CATEGORIES = [('Filters', 'Braking'), ('Engine', 'Cooling'), ('Cooling', 'Electrical'), ('Body', 'Engine')]


def generate(root=Path('data'), seed=42, n_skus=200, days=730):
    root = Path(root)
    if n_skus < 4 or days < 365:
        raise ValueError('At least four SKUs and 365 days required')
    for folder in ['raw', 'processed', 'mutated_schemas']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    records, metadata, series = [], [], {}
    for i in range(n_skus):
        r = i % 4
        target = REGIMES[r]
        level = rng.uniform(4, 20) if r < 2 else rng.uniform(2, 8)
        for _ in range(1000):
            occurrence = rng.random(days) < ([.96, .94, .17, .12][r])
            seasonal = 1 + .12 * np.sin(2 * np.pi * np.arange(days) / 365 + i)
            sizes = rng.lognormal(np.log(level) - ([.18, 1.15, .18, 1.35][r]**2) / 2,
                                  [.18, 1.15, .18, 1.35][r], days)
            y = occurrence * np.maximum(1, np.rint(sizes * seasonal)).astype(int)
            if classify(y[:int(.8 * days)])['regime'] == target and classify(y)['regime'] == target:
                break
        else:
            raise RuntimeError('Unable to sample requested regime')
        sku = f'{["TOY", "HON", "NIS"][i % 3]}-{10000+i}-{i % 17:03d}'
        lead = int(rng.choice([7, 14, 21, 28]))
        # Static opening stock scenario independent of future realised demand.
        stock = int(level * [.96, .94, .17, .12][r] * lead)
        cost = round(float(rng.uniform(400, 4000) if r < 2 else rng.uniform(5000, 90000)), 2)
        variant = (i // 4) % 2
        item = dict(part_id=sku, part_name=PARTS[r][variant], category=CATEGORIES[r][variant],
                    current_stock=stock, lead_time_days=lead, unit_cost=cost)
        for date, quantity in zip(dates[y > 0], y[y > 0]):
            records.append(dict(item, transaction_date=date.strftime('%Y-%m-%d'), quantity_sold=int(quantity)))
        metadata.append(dict(item, intended_regime=target, **classify(y[:int(.8 * days)])))
        series[sku] = y
    transactions = pd.DataFrame(records)[FIELDS].sort_values(['transaction_date', 'part_id'])
    transactions.to_csv(root / 'raw' / 'automotive_transactions.csv', index=False)
    pd.DataFrame(metadata).to_csv(root / 'raw' / 'sku_metadata.csv', index=False)
    dense = pd.DataFrame(series, index=dates)
    dense.index.name = 'date'
    dense.to_csv(root / 'processed' / 'daily_demand.csv')
    manifest = {'seed': seed, 'n_skus': n_skus, 'days': days, 'start': str(dates[0].date()),
                'end': str(dates[-1].date()), 'frequency': 'D', 'mutations': []}
    tier1 = [
        ['OEM_Part_No', 'Sale_Dt', 'Qty_Sold', 'Description', 'Part_Group', 'Stock_On_Hand', 'Lead_Days', 'Unit_Price'],
        ['Item_Code', 'Invoice_Date', 'Issued_Qty', 'Item_Description', 'Vehicle_System', 'Stock_Balance', 'Supplier_LT', 'Cost_LKR']]
    tier2 = [
        ['Chas_SKU', 'DMS_Post_Dt', 'GRN_Iss_Qty', 'Nomenclature', 'Veh_Sys', 'SOH_Bal', 'L_Time', 'W_Cost'],
        ['EPC_Ref', 'GI_Budat', 'Mvt_261_Units', 'EPC_Text', 'Matkl', 'Labst', 'Plifz', 'Verpr']]
    for tier in range(1, 4):
        for j in range(10):
            # Vary row samples, order, names and irrelevant distractors. No regime label is exposed.
            sample = transactions.sample(n=min(1200, len(transactions)), random_state=seed + tier * 100 + j).copy()
            if tier < 3:
                names = (tier1 if tier == 1 else tier2)[j % 2]
                names = [name if j < 2 else f'{name}_{j}' for name in names]
            else:
                names = ['COL_01', 'DT_FLD_A', 'NUM_VAL_1', 'TXT_B', 'TXT_C', 'NUM_VAL_2', 'NUM_VAL_3', 'NUM_VAL_4']
                # Random assignment prevents positional or numeric suffix shortcuts.
                names = list(rng.permutation(names))
                if j > 0:
                    names = [f'{name}_{j}' for name in names]
            truth = dict(zip(FIELDS, names))
            sample = sample.rename(columns=truth)
            sample['Export_Batch' if tier < 3 else f'COL_X_{j}'] = f'BATCH-{j:03d}'
            sample = sample[list(rng.permutation(sample.columns))]
            path = f'mutated_schemas/tier_{tier}_{j+1:02d}.csv'
            sample.to_csv(root / path, index=False)
            manifest['mutations'].append({'file': path, 'tier': tier, 'truth': truth})
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.seed), indent=2))


if __name__ == '__main__':
    main()
