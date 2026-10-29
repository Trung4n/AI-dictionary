"""Per-drawing statistics over the whole parquet, computed in parallel.

Run once from the repo root (takes a few minutes):

    PYTHONPATH=src .venv/Scripts/python -m quickdraw.stats
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .io import DERIVED_DIR, META_COLUMNS, PARQUET_PATH

STATS_PATH = DERIVED_DIR / "drawing_stats.parquet"
BATCH_SIZE = 50_000

STAT_COLUMNS = [
    "n_strokes", "n_points", "min_stroke_points", "max_stroke_points",
    "n_single_point_strokes", "xmin", "ymin", "xmax", "ymax",
    "n_out_of_range", "ink_length", "first_stroke_ink_share", "drawing_hash",
]


def drawing_hash(s: str) -> int:
    """Stable 64-bit hash of the raw drawing string (Python's hash() is salted)."""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


def drawing_stats(s: str) -> tuple:
    strokes = json.loads(s)
    lens = [len(xs) for xs, _ in strokes]
    xs_all = [x for xs, _ in strokes for x in xs]
    ys_all = [y for _, ys in strokes for y in ys]
    inks = []
    for xs, ys in strokes:
        inks.append(sum(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(len(xs) - 1)))
    ink = sum(inks)
    out_of_range = sum(1 for v in xs_all if not 0 <= v <= 255) + sum(1 for v in ys_all if not 0 <= v <= 255)
    return (
        len(strokes), sum(lens), min(lens), max(lens), sum(1 for n in lens if n == 1),
        min(xs_all), min(ys_all), max(xs_all), max(ys_all),
        out_of_range, ink, inks[0] / ink if ink > 0 else float("nan"), drawing_hash(s),
    )


def _batch_stats(drawings: list[str]) -> list[tuple]:
    return [drawing_stats(s) for s in drawings]


def compute_stats(path: Path = PARQUET_PATH, workers: int = 4) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    meta = pf.read(columns=META_COLUMNS).to_pandas()
    batches = (b.column("drawing").to_pylist() for b in pf.iter_batches(BATCH_SIZE, columns=["drawing"]))
    rows = []
    # Keep only a few batches in flight: the machine may have little RAM, and
    # Executor.map would submit (and pickle) every batch up front.
    with ProcessPoolExecutor(workers) as ex:
        pending = deque()
        for batch in batches:
            pending.append(ex.submit(_batch_stats, batch))
            if len(pending) >= 2 * workers:
                rows.extend(pending.popleft().result())
                print(f"\r{len(rows):,} / {len(meta):,}", end="", flush=True)
        while pending:
            rows.extend(pending.popleft().result())
    print(f"\r{len(rows):,} / {len(meta):,}")
    stats = pd.DataFrame(rows, columns=STAT_COLUMNS)
    df = pd.concat([meta, stats], axis=1)
    df.insert(0, "row", range(len(df)))
    return df


def load_stats(path: Path = STATS_PATH) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        compute_stats().to_parquet(path, index=False)
    return pd.read_parquet(path)


if __name__ == "__main__":
    import time

    t = time.time()
    df = load_stats()
    print(df.shape, f"{time.time() - t:.0f}s")
