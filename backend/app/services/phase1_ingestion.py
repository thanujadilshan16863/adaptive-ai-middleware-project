import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

from backend.app.config import settings


def ingest(content: bytes, filename: str) -> pd.DataFrame:
    if not content or len(content) > settings.max_upload_mb * 1024**2:
        raise ValueError('Empty file or upload exceeds size limit')
    suffix = Path(filename).suffix.lower()
    if suffix == '.csv':
        text = content.decode('utf-8-sig')
        headers = next(csv.reader(io.StringIO(text)))
        if len(headers) != len(set(headers)):
            raise ValueError('Duplicate CSV headers are ambiguous')
        frame = pd.read_csv(io.StringIO(text), nrows=settings.max_rows + 1)
    elif suffix == '.xlsx':
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if sum(i.file_size for i in archive.infolist()) > 200 * 1024**2:
                raise ValueError('Expanded workbook exceeds limit')
        frame = pd.read_excel(io.BytesIO(content), nrows=settings.max_rows + 1)
    else:
        raise ValueError('Only UTF-8 CSV and XLSX are supported')
    frame = frame.dropna(how='all').dropna(axis=1, how='all')
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.empty or len(frame) > settings.max_rows or not 3 <= len(frame.columns) <= 100:
        raise ValueError('Invalid table dimensions')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names')
    return frame


def profile(frame: pd.DataFrame) -> list[dict]:
    return [dict(name=c, dtype=str(frame[c].dtype), null_ratio=float(frame[c].isna().mean()),
                 unique_ratio=float(frame[c].nunique() / len(frame)),
                 samples=frame[c].dropna().astype(str).drop_duplicates().head(3).tolist())
            for c in frame]
