import numpy as np
import pandas as pd

from backend.app.config import settings
from backend.app.schemas import REQUIRED


def classify(values):
    y = np.asarray(values, dtype=float)
    if not len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Demand must be finite, nonnegative and nonempty')
    positive = y[y > 0]
    adi = len(y) / len(positive) if len(positive) else float('inf')
    cv2 = float(positive.var(ddof=1) / positive.mean()**2) if len(positive) > 1 else None
    regime = ('All-zero' if not len(positive) else 'Insufficient' if cv2 is None else
              ('Smooth' if cv2 < .49 else 'Erratic') if adi < 1.32 else
              ('Intermittent' if cv2 < .49 else 'Lumpy'))
    return {'ADI': adi, 'CV2': cv2, 'regime': regime}


def canonicalize(frame, mapping, min_days=365):
    mapping = mapping.model_dump()
    if any(mapping[k] is None for k in REQUIRED):
        raise ValueError('Required mapping unresolved; supply a reviewed mapping')
    if any(v not in frame.columns for v in mapping.values() if v):
        raise ValueError('Mapping references unknown columns')
    df = frame.rename(columns={v: k for k, v in mapping.items() if v})[[k for k, v in mapping.items() if v]].copy()
    if df[REQUIRED].isna().any().any():
        raise ValueError('Required fields contain nulls')
    df['part_id'] = df.part_id.astype(str).str.strip()
    if df.part_id.eq('').any() or df.part_id.nunique() > settings.max_skus:
        raise ValueError('Invalid SKU identifiers or too many SKUs')
    df['transaction_date'] = pd.to_datetime(df.transaction_date, errors='raise', utc=True).dt.tz_convert(None).dt.normalize()
    defaults = {'part_name': 'Unknown', 'category': 'Unknown', 'current_stock': 0., 'lead_time_days': 14, 'unit_cost': 0.}
    for col, default in defaults.items():
        if col not in df:
            df[col] = default
        df[col] = df[col].fillna(default)
    for col in ['quantity_sold', 'current_stock', 'lead_time_days', 'unit_cost']:
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f'{col} must be finite and nonnegative')
    if ((df.lead_time_days < 1) | (df.lead_time_days > 365) | (df.lead_time_days % 1 != 0)).any():
        raise ValueError('Lead time must be an integer from 1 to 365 days')
    if (df.transaction_date.max() - df.transaction_date.min()).days + 1 < min_days:
        raise ValueError(f'At least {min_days} calendar days of history required')
    return df.sort_values(['part_id', 'transaction_date'], kind='stable')


def zero_fill(df, start=None, end=None, frequency='D'):
    if frequency not in ('D', '7D'):
        raise ValueError('Frequency must be D or 7D')
    start = pd.Timestamp(start if start is not None else df.transaction_date.min())
    end = pd.Timestamp(end if end is not None else df.transaction_date.max())
    if end < start or start > df.transaction_date.min() or end < df.transaction_date.max():
        raise ValueError('Observation window must contain all transactions')
    grid = pd.date_range(start, end, freq='D')
    if len(grid) * df.part_id.nunique() > 10000000:
        raise ValueError('Temporal grid exceeds ten million cells')
    pivot = df.groupby(['transaction_date', 'part_id']).quantity_sold.sum().unstack(fill_value=0)
    pivot = pivot.reindex(grid, fill_value=0).fillna(0).astype(float)
    pivot.index.name = 'date'
    if frequency == '7D':
        # Retain the final partial week, explicitly; 730 days produce 105 bins.
        pivot = pivot.resample('7D', origin=start).sum()
    return pivot
