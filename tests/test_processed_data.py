"""Checks on the packed dataset written by ``python -m quickdraw.prepare``.
Skipped when it has not been built."""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from quickdraw.io import PARQUET_PATH, parse_drawing
from quickdraw.prepare import N_TEST, N_VAL, PROCESSED_DIR, SPLITS
from quickdraw.transforms import EVAL_FRACS

pytestmark = pytest.mark.skipif(not (PROCESSED_DIR / "manifest.json").exists(),
                                reason="processed dataset not built")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((PROCESSED_DIR / "manifest.json").read_text())


@pytest.fixture(scope="module")
def metas():
    return {s: pd.read_parquet(PROCESSED_DIR / f"{s}_meta.parquet") for s in SPLITS}


@pytest.fixture(scope="module")
def splits():
    from quickdraw.dataset import QuickDrawSplit
    return {s: QuickDrawSplit(s) for s in SPLITS}


def test_splits_are_disjoint(metas):
    keys = [set(m["key_id"]) for m in metas.values()]
    rows = [set(m["row"]) for m in metas.values()]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not keys[i] & keys[j] and not rows[i] & rows[j]


def test_every_row_accounted_for(metas, manifest):
    n = sum(len(m) for m in metas.values())
    assert n + manifest["rows_dropped"] == manifest["rows_in_source"] == 3_000_000
    dropped = set(manifest["dropped_key_ids"])
    assert all(not dropped & set(m["key_id"]) for m in metas.values())


def test_eval_splits_are_balanced(metas):
    assert (metas["val"]["word"].value_counts() == N_VAL).all()
    assert (metas["test"]["word"].value_counts() == N_TEST).all()
    assert metas["val"]["word"].nunique() == metas["test"]["word"].nunique() == 100


def test_no_duplicate_drawing_across_splits(metas):
    from quickdraw.stats import load_stats
    h = load_stats()[["row", "drawing_hash"]].set_index("row")["drawing_hash"]
    hashes = [set(h.loc[m["row"]]) for m in metas.values()]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not hashes[i] & hashes[j]


def test_offsets_consistent(splits, metas):
    for s, ds in splits.items():
        assert ds.offsets[0] == 0 and ds.offsets[-1] == len(ds.xy) == len(ds.end)
        assert np.array_equal(np.diff(ds.offsets), metas[s]["n_points"].to_numpy())
        # every drawing ends with a pen lift, and pen lifts = strokes
        assert ds.end[ds.offsets[1:] - 1].all()
        assert int(ds.end.sum()) == int(metas[s]["n_strokes"].sum())


def test_packed_drawings_match_source(splits):
    """Decode random packed drawings and compare with the original JSON."""
    rng = np.random.default_rng(0)
    table = pq.read_table(PARQUET_PATH, columns=["drawing"]).column("drawing")
    for s, ds in splits.items():
        for i in rng.choice(len(ds), 300, replace=False):
            pts, end, label = ds[i]
            strokes = parse_drawing(table[int(ds.meta["row"].iloc[i])].as_py())
            ref = np.array([(x, y) for xs, ys in strokes for x, y in zip(xs, ys)])
            assert np.array_equal(pts, ref)
            assert np.array_equal(np.flatnonzero(end) + 1, np.cumsum([len(xs) for xs, _ in strokes]))
            assert label == ds.meta["label"].iloc[i]


def test_split_is_reproducible(manifest):
    from quickdraw.prepare import make_split, split_digest
    from quickdraw.stats import load_stats
    kept, dropped = make_split(load_stats(), seed=manifest["seed"])
    assert split_digest(kept) == manifest["split_digest"]
    assert sorted(int(k) for k in dropped["key_id"]) == manifest["dropped_key_ids"]


def test_manifest_hashes(manifest):
    import hashlib
    for name, digest in manifest["files"].items():
        h = hashlib.sha256((PROCESSED_DIR / name).read_bytes()).hexdigest()
        assert h == digest, name


def test_seq_stats_fitted_on_train(manifest):
    st = json.loads((PROCESSED_DIR / "seq_stats.json").read_text())
    assert st["fitted_on"] == "train" and st == manifest["seq_stats"]
    assert all(v > 0 for v in st["delta_std"])


def test_views_produce_valid_samples(splits):
    from quickdraw.dataset import RasterView, SequenceView
    from quickdraw.transforms import AugmentConfig

    train = RasterView(splits["train"], frac="random", augment=AugmentConfig(), seed=1)
    a, b = train[123], train[123]
    assert np.array_equal(a[0], b[0])           # reproducible
    train.set_epoch(1)
    assert not np.array_equal(train[123][0], a[0])  # changes with epoch
    for f in EVAL_FRACS:
        img, _ = RasterView(splits["val"], frac=f)[7]
        seq, n, _ = SequenceView(splits["val"], frac=f)[7]
        assert img.shape == (64, 64) and img.max() > 0
        assert seq.shape == (256, 3) and 1 <= n <= 256
    with pytest.raises(ValueError):
        RasterView(splits["test"], frac="random")


def test_sequence_view_resample_option(splits):
    from quickdraw.dataset import SequenceView, load_resampled_stats
    from quickdraw.transforms import RESAMPLE_STEP

    plain = SequenceView(splits["val"], frac=0.25)
    res = SequenceView(splits["val"], frac=0.25, resample_step=RESAMPLE_STEP)
    assert res.delta_std == load_resampled_stats(RESAMPLE_STEP) != plain.delta_std
    seq, n, label = res[7]
    assert seq.shape == (256, 3) and 1 <= n <= 256 and label == plain[7][2]
    with pytest.raises(ValueError):
        SequenceView(splits["val"], resample_step=RESAMPLE_STEP + 1)
