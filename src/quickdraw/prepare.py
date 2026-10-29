"""Clean, split and pack the dataset (decisions from notebooks/01_eda.ipynb).

    PYTHONPATH=src .venv/Scripts/python -m quickdraw.prepare

Steps
1. Drop every drawing whose exact stroke data also appears under another class
   (39 groups / 99 rows, all 2-point lines). Nothing else is removed; tiny,
   dot-only, badly normalised and ``recognized=False`` drawings are kept and flagged.
2. Split per class with a fixed seed: exactly ``N_VAL`` validation and ``N_TEST``
   test drawings per class, the rest train. Identical drawings (same hash) are
   assigned as one unit, so copies can never land in different splits.
3. Pack each split into memory-mappable arrays under ``dataset/processed/``:
   ``{split}_xy.npy`` (uint8, P x 2, raw coordinates), ``{split}_end.npy`` (bool,
   pen lift after the point), ``{split}_offsets.npy`` (int64, N + 1) and
   ``{split}_meta.parquet``.
4. Fit the stroke-3 delta scale on the *training* split only (``seq_stats.json``).
5. Write ``manifest.json`` with parameters, counts and sha256 of every output.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .io import DATASET_DIR, PARQUET_PATH, ROOT
from .stats import load_stats

PROCESSED_DIR = DATASET_DIR / "processed"
REPORT_DIR = ROOT / "reports" / "preprocessing"
SPLITS = ("train", "val", "test")
SPLIT_VERSION = "v1"
SEED = 42
N_VAL = 3000
N_TEST = 3000
BATCH_SIZE = 50_000
WORKERS = 4

META_OUT = ["key_id", "row", "label", "word", "recognized", "countrycode", "n_points", "n_strokes",
            "tiny", "dots_only", "severe_nonconform"]


# ---------------------------------------------------------------- split

def make_split(stats: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns ``(kept, dropped)``; ``kept`` has a ``split`` column."""
    n_cls = stats.groupby("drawing_hash")["word"].transform("nunique")
    dropped = stats[n_cls > 1].copy()
    kept = stats[n_cls == 1].copy()

    rng = np.random.default_rng(seed)
    kept["split"] = ""
    for word in sorted(kept["word"].unique()):
        cls = kept[kept["word"] == word]
        # Unit of assignment = group of identical drawings, ordered by its smallest key_id.
        units = cls.groupby("drawing_hash")["key_id"].min().sort_values().index.to_numpy()
        units = units[rng.permutation(len(units))]
        size = cls.groupby("drawing_hash").size().reindex(units).to_numpy()
        cum = np.cumsum(size)
        unit_split = np.where(cum <= N_VAL, "val", np.where(cum <= N_VAL + N_TEST, "test", "train"))
        kept.loc[cls.index, "split"] = cls["drawing_hash"].map(dict(zip(units, unit_split)))
    return kept, dropped


def split_digest(kept: pd.DataFrame) -> str:
    """sha256 of the sorted ``key_id,split`` lines: identifies the split itself,
    independent of file formats."""
    lines = kept.sort_values("key_id")[["key_id", "split"]].astype(str).agg(",".join, axis=1)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ---------------------------------------------------------------- packing

def _pack_batch(args):
    """Parse a batch of JSON drawings. Returns packed points for all rows and the
    delta sums of rows flagged as training, after normalising each drawing."""
    drawings, is_train = args
    xy, end = [], []
    s = np.zeros(2)
    s2 = np.zeros(2)
    count = 0
    for text, train in zip(drawings, is_train):
        strokes = json.loads(text)
        p = np.array([(x, y) for xs, ys in strokes for x, y in zip(xs, ys)], dtype=np.int64)
        e = np.zeros(len(p), dtype=bool)
        e[np.cumsum([len(xs) for xs, _ in strokes]) - 1] = True
        xy.append(p.astype(np.uint8))
        end.append(e)
        if train and len(p) > 1:
            lo = p.min(axis=0)
            side = (p.max(axis=0) - lo).max()
            d = np.diff(p, axis=0) * (255.0 / side if side > 0 else 1.0)
            s += d.sum(axis=0)
            s2 += (d ** 2).sum(axis=0)
            count += len(d)
    return np.concatenate(xy), np.concatenate(end), s, s2, count


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(out_dir: Path = PROCESSED_DIR, workers: int = WORKERS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = load_stats()
    if not (stats["n_out_of_range"] == 0).all():
        raise ValueError("coordinates outside 0..255 cannot be stored as uint8")
    stats["tiny"] = stats["n_points"] <= 3
    stats["dots_only"] = stats["ink_length"] == 0
    offset = stats[["xmin", "ymin"]].max(axis=1)
    stats["severe_nonconform"] = (offset > 2) | (stats[["xmax", "ymax"]].max(axis=1) < 253)

    kept, dropped = make_split(stats)
    split_code = np.full(len(stats), -1, dtype=np.int8)
    for i, name in enumerate(SPLITS):
        split_code[kept.loc[kept["split"] == name, "row"].to_numpy()] = i
    n_points = stats["n_points"].to_numpy()

    # Rows of one split stay in file order, so every batch writes one contiguous block per split.
    arrays, cursor = {}, {}
    for i, name in enumerate(SPLITS):
        sel = split_code == i
        meta = stats.loc[sel, META_OUT].reset_index(drop=True)
        meta.to_parquet(out_dir / f"{name}_meta.parquet", index=False)
        offsets = np.concatenate([[0], np.cumsum(n_points[sel])]).astype(np.int64)
        np.save(out_dir / f"{name}_offsets.npy", offsets)
        total = int(offsets[-1])  # a NumPy int would be written into the .npy header as "np.int64(...)"
        arrays[name] = (
            np.lib.format.open_memmap(out_dir / f"{name}_xy.npy", "w+", np.uint8, (total, 2)),
            np.lib.format.open_memmap(out_dir / f"{name}_end.npy", "w+", np.bool_, (total,)),
        )
        cursor[name] = 0

    pf = pq.ParquetFile(PARQUET_PATH)
    s, s2, count, start = np.zeros(2), np.zeros(2), 0, 0

    def write(result, rows):
        nonlocal s, s2, count
        xy, end, bs, bs2, bc = result
        s, s2, count = s + bs, s2 + bs2, count + bc
        point_split = np.repeat(split_code[rows], n_points[rows])
        if len(point_split) != len(xy):
            raise RuntimeError("parsed point count differs from drawing_stats")
        for i, name in enumerate(SPLITS):
            m = point_split == i
            k = int(m.sum())
            arrays[name][0][cursor[name]:cursor[name] + k] = xy[m]
            arrays[name][1][cursor[name]:cursor[name] + k] = end[m]
            cursor[name] += k

    with ProcessPoolExecutor(workers) as ex:
        pending = deque()
        for batch in pf.iter_batches(BATCH_SIZE, columns=["drawing"]):
            rows = np.arange(start, start + batch.num_rows)
            start += batch.num_rows
            pending.append((ex.submit(_pack_batch, (batch.column("drawing").to_pylist(), split_code[rows] == 0)), rows))
            if len(pending) >= 2 * workers:
                fut, r = pending.popleft()
                write(fut.result(), r)
                print(f"\r{r[-1] + 1:,} / {len(stats):,}", end="", flush=True)
        while pending:
            fut, r = pending.popleft()
            write(fut.result(), r)
    print(f"\r{len(stats):,} / {len(stats):,}")
    for name in SPLITS:
        if cursor[name] != arrays[name][0].shape[0]:
            raise RuntimeError(f"{name}: wrote {cursor[name]} points, expected {arrays[name][0].shape[0]}")
        for a in arrays[name]:
            a.flush()
    del arrays

    mean = s / count
    std = np.sqrt(s2 / count - mean ** 2)
    seq_stats = {"fitted_on": "train", "n_deltas": int(count),
                 "delta_mean": mean.round(6).tolist(), "delta_std": std.round(6).tolist(),
                 "note": "deltas between consecutive points of full drawings normalised to long side 255"}
    (out_dir / "seq_stats.json").write_text(json.dumps(seq_stats, indent=2))

    counts = kept.groupby(["split"])["row"].size().reindex(list(SPLITS)).astype(int).to_dict()
    per_class = kept.groupby(["word", "split"]).size().unstack()[list(SPLITS)]
    manifest = {
        "split_version": SPLIT_VERSION,
        "seed": SEED,
        "n_val_per_class": N_VAL,
        "n_test_per_class": N_TEST,
        "source": PARQUET_PATH.name,
        "source_sha256": _sha256(PARQUET_PATH),
        "rows_in_source": int(len(stats)),
        "rows_dropped": int(len(dropped)),
        "dropped_reason": "identical stroke data labelled with more than one class",
        "dropped_key_ids": sorted(int(k) for k in dropped["key_id"]),
        "rows_per_split": counts,
        "train_per_class_min_max": [int(per_class["train"].min()), int(per_class["train"].max())],
        "split_digest": split_digest(kept),
        "seq_stats": seq_stats,
        "files": {p.name: _sha256(p) for p in sorted(out_dir.iterdir()) if p.name != "manifest.json"},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    per_class.to_csv(REPORT_DIR / "split_per_class.csv")
    return manifest


if __name__ == "__main__":
    import time

    t = time.time()
    m = prepare()
    print(json.dumps({k: m[k] for k in ("rows_dropped", "rows_per_split", "split_digest")}, indent=2))
    print(f"{time.time() - t:.0f}s")
