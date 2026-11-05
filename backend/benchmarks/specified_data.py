"""Read the supplied four-file evaluation dataset without altering source files."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from backend.app.schemas import SchemaMappingResult
from backend.app.services.phase3_normalizer import canonicalize, zero_fill, classify

MASTER = 'canonical_automotive_inventory_master.csv'
EXPORTS = ['erp_export_tier1_synonyms.csv', 'erp_export_tier2_jargon.csv', 'erp_export_tier3_cryptic.csv']
MASTER_MAPPING = {'part_id': 'sku', 'transaction_date': 'date', 'quantity_sold': 'sold_qty',
                  'part_name': 'title', 'current_stock': 'stock_bal', 'unit_cost': 'unit_price',
                  'category': None, 'lead_time_days': None}


def load_specified_data(folder):
    folder = Path(folder)
    master = pd.read_csv(folder / MASTER)
    mapping = SchemaMappingResult(**MASTER_MAPPING)
    canonical = canonicalize(master, mapping)
    grid = zero_fill(canonical)
    metadata = canonical.groupby('part_id', sort=True).tail(1).set_index('part_id')
    metadata['regime'] = [classify(grid[c].values[:int(.8 * len(grid))])['regime'] for c in metadata.index]
    mutations = []
    for tier, filename in enumerate(EXPORTS, 1):
        export = pd.read_csv(folder / filename)
        truth = {}
        # Ground truth is established from exact full-column equality to the master,
        # independent of all evaluated mappers. Ambiguous/nonidentical files require a sidecar.
        sidecar = folder / 'schema_ground_truth.json'
        supplied = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else {}
        for field, column in MASTER_MAPPING.items():
            if column is None:
                truth[field] = None
                continue
            candidates = [c for c in export if export[c].equals(master[column])]
            truth[field] = supplied.get(filename, {}).get(field)
            if truth[field] is None:
                if len(candidates) != 1:
                    raise ValueError(f'{filename}: cannot establish independent truth for {field}; provide schema_ground_truth.json')
                truth[field] = candidates[0]
            if truth[field] not in export:
                raise ValueError(f'{filename}: ground truth references an absent column')
        SchemaMappingResult(**truth)
        mutations.append({'tier': tier, 'file': filename, 'truth': truth})
    provenance = {'source': str(folder.resolve()), 'n_skus': grid.shape[1], 'days': len(grid),
        'files_sha256': {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in [MASTER, *EXPORTS]},
        'assumptions': ['Continuous assortment: absent SKU-days are zeros',
            'Missing lead_time_days set to 14 days for a scenario, not measured supplier lead times',
            'Missing category set to Unknown',
            'unit_price is a scenario value proxy, not verified acquisition cost or currency',
            'Schema truth uses exact full-column equality or a supplied sidecar; never mapper predictions']}
    return {'mutations': mutations}, grid, metadata, canonical, provenance


def workload(grid, metadata, size):
    """Dense balanced cyclic SKU replication for scaling only, never accuracy scoring."""
    groups = [list(g.index) for _, g in metadata.groupby('regime', sort=True)]
    ordered = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
    frames = []
    for i in range(size):
        source = ordered[i % len(ordered)]
        item = metadata.loc[source]
        frames.append(pd.DataFrame({'part_id': f'STRESS-{i:04d}', 'transaction_date': grid.index,
            'quantity_sold': grid[source].values, 'part_name': item.part_name, 'category': item.category,
            'current_stock': item.current_stock, 'lead_time_days': item.lead_time_days, 'unit_cost': item.unit_cost}))
    return pd.concat(frames, ignore_index=True)
