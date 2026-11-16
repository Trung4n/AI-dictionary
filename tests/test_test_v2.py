"""Tests for the test_v2 builder, on a tiny fake competition folder and split v1."""
import json

import numpy as np
import pandas as pd
import pytest

from quickdraw_exp.test_v2 import build, pack, packed_text, read_class, text_hash

WORDS = [f"class {i:02d}" for i in range(100)]


def _drawing(rng) -> str:
    strokes = []
    for _ in range(rng.integers(1, 4)):
        n = int(rng.integers(2, 6))
        strokes.append([rng.integers(0, 256, n).tolist(), rng.integers(0, 256, n).tolist()])
    return json.dumps(strokes)  # with spaces, as in the Kaggle files


def _fake(tmp_path, rows_per_class=12):
    rng = np.random.default_rng(0)
    root, kaggle = tmp_path / "processed", tmp_path / "comp"
    (kaggle / "train_simplified").mkdir(parents=True)
    root.mkdir()
    used, key = {}, 1
    for split in ("train", "val", "test"):
        xy, end, offsets, meta = [], [], [0], []
        for label, word in enumerate(WORDS):
            text = _drawing(rng)
            p, e = pack(text)
            xy.append(p)
            end.append(e)
            offsets.append(offsets[-1] + len(p))
            meta.append({"key_id": key, "label": label, "word": word})
            used.setdefault(word, []).append((key, text))
            key += 1
        np.save(root / f"{split}_xy.npy", np.concatenate(xy))
        np.save(root / f"{split}_end.npy", np.concatenate(end))
        np.save(root / f"{split}_offsets.npy", np.array(offsets, np.int64))
        pd.DataFrame(meta).to_parquet(root / f"{split}_meta.parquet", index=False)
    (root / "manifest.json").write_text(json.dumps({"dropped_key_ids": [999999]}))
    shared = _drawing(rng)  # appears in two classes: a label conflict
    for label, word in enumerate(WORDS):
        rows = [{"countrycode": "US", "drawing": t, "key_id": k, "recognized": "True", "word": word} for k, t in used[word][:1]]
        rows.append({"countrycode": "US", "drawing": used[WORDS[(label + 1) % 100]][1][1].replace(", ", ","),
                     "key_id": 10_000 + label, "recognized": "True", "word": word})  # repeats a v1 drawing of another class
        rows.append({"countrycode": "IN", "drawing": _drawing(rng), "key_id": 20_000 + label, "recognized": "True", "word": word})
        if label < 2:
            rows.append({"countrycode": "US", "drawing": shared, "key_id": 30_000 + label, "recognized": "True", "word": word})
        for j in range(rows_per_class):
            rows.append({"countrycode": "NA" if j == 0 else "VN", "drawing": _drawing(rng), "key_id": 100_000 + 100 * label + j,
                         "recognized": "False" if j % 2 else "True", "word": word})
        pd.DataFrame(rows).to_csv(kaggle / "train_simplified" / f"{word}.csv", index=False)
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps({"countries": ["IN", "IT", "KR", "SA", "TH"]}))
    return kaggle, root, {"per_class": 5, "reserve": 3, "seed": 2027, "holdout_file": str(holdout)}


def test_packed_text_matches_the_kaggle_text():
    text = "[[[1, 2, 3], [4, 5, 6]], [[7], [8]]]"
    assert packed_text(*pack(text)) == text.replace(" ", "")
    assert text_hash(text) == text_hash(text.replace(" ", ""))


def test_build_keeps_only_fresh_drawings(tmp_path):
    kaggle, root, spec = _fake(tmp_path)
    m = build(kaggle, root, tmp_path / "out", spec)
    meta = pd.read_parquet(tmp_path / "out" / "test_v2_meta.parquet")
    assert len(meta) == 500 and m["drawings"] == 500
    assert (meta.groupby("label").size() == 5).all()
    assert (meta["key_id"] >= 100_000).all()  # no used key_id, no repeat of v1, no held-out country, no conflict
    t = m["totals"]
    assert t["used_key_id"] == 100 and t["held_out_country"] == 100 and t["repeats_split_v1"] == 100
    offsets = np.load(tmp_path / "out" / "test_v2_offsets.npy")
    assert offsets[-1] == len(np.load(tmp_path / "out" / "test_v2_xy.npy")) and len(offsets) == 501
    again = build(kaggle, root, tmp_path / "again", spec)
    assert again["key_id_digest"] == m["key_id_digest"]


def test_label_conflicts_are_dropped(tmp_path):
    kaggle, root, spec = _fake(tmp_path)
    spec["reserve"] = 20  # the whole class is in the pool
    build(kaggle, root, tmp_path / "out", spec)
    meta = pd.read_parquet(tmp_path / "out" / "test_v2_meta.parquet")
    assert not meta["key_id"].isin([30_000, 30_001]).any()


def test_too_few_drawings_is_an_error(tmp_path):
    kaggle, root, spec = _fake(tmp_path, rows_per_class=3)
    with pytest.raises(ValueError):
        build(kaggle, root, tmp_path / "out", spec)


def test_namibia_is_a_country_and_empty_is_missing(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("countrycode,drawing,key_id,recognized,word,timestamp\nNA,\"[[[1],[2]]]\",5,True,x,t\n,\"[[[1],[2]]]\",6,False,x,t\n")
    df = read_class(path)
    assert df["countrycode"].iloc[0] == "NA" and pd.isna(df["countrycode"].iloc[1])
    assert df["recognized"].tolist() == [True, False]
