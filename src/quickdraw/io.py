"""Loading the parquet subset and parsing drawings.

A drawing is stored as a JSON string in the Kaggle "simplified" format:
``[[[x0, x1, ...], [y0, y1, ...]], ...]`` with one ``[xs, ys]`` pair per stroke,
strokes in drawing order.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
PARQUET_PATH = DATASET_DIR / "quickdraw_100c_30k.parquet"
CLASSES_PATH = DATASET_DIR / "quickdraw_100c_30k_classes.txt"
SELECTION_DIR = DATASET_DIR / "class_selection"
DERIVED_DIR = DATASET_DIR / "derived"

META_COLUMNS = ["key_id", "label", "word", "recognized", "countrycode"]

Stroke = tuple[list[int], list[int]]


def parse_drawing(s: str) -> list[Stroke]:
    return [(xs, ys) for xs, ys in json.loads(s)]


def load_meta(path: Path = PARQUET_PATH) -> pd.DataFrame:
    """All columns except ``drawing``."""
    return pq.read_table(path, columns=META_COLUMNS).to_pandas()


def load_classes(path: Path = CLASSES_PATH) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def load_drawings(rows, path: Path = PARQUET_PATH) -> list[list[Stroke]]:
    """Parsed drawings for the given row positions, returned in the order given.
    Reads one row group at a time to keep memory bounded."""
    rows = list(rows)
    pf = pq.ParquetFile(path)
    out: dict[int, list[Stroke]] = {}
    start = 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        local = sorted({r - start for r in rows if start <= r < start + n})
        if local:
            col = pf.read_row_group(g, columns=["drawing"]).column("drawing")
            for r, s in zip(local, col.take(local).to_pylist()):
                out[start + r] = parse_drawing(s)
            del col
        start += n
    return [out[r] for r in rows]
