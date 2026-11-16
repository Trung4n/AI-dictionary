"""Build test_v2: a fresh test set from Quick, Draw! drawings the project never used.
CPU only. It writes drawings and prints counts; no model sees it here.

    PYTHONPATH=src python -m quickdraw_exp.test_v2 --kaggle <folder holding train_simplified/> \
        --out <folder> [--config configs/test_v2.yaml]

The 3,000,000 drawings of split v1 were drawn at random from the Kaggle competition's
``train_simplified`` files (30,000 per class). Many more drawings of the same 100
classes remain there. For every class, test_v2 keeps the drawings that

* have a key_id that is not in split v1 (train, validation, test, or the 99 rows
  dropped as label conflicts),
* are not from a held-out country (those form a separate test),
* do not repeat, stroke for stroke, any drawing of split v1 (any class),
* do not repeat a drawing of another class among the candidates (label conflicts,
  as in split v1), and are the first of identical copies within their class,

then draws ``per_class`` of them at random with a fixed seed. Strokes are compared
after removing spaces from the stored JSON, and split v1 is rebuilt into the same
text from its packed arrays, so formatting cannot hide a duplicate.

The output has the layout of split v1 (``test_v2_xy.npy``, ``test_v2_end.npy``,
``test_v2_offsets.npy``, ``test_v2_meta.parquet``) and a ``manifest_test_v2.json``
with every count and the sha256 of every file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SPLIT = "test_v2"
USED_SPLITS = ("train", "val", "test")


def text_hash(s: str) -> int:
    """64-bit hash of a drawing's JSON with the spaces removed."""
    return int.from_bytes(hashlib.blake2b(s.replace(" ", "").encode(), digest_size=8).digest(), "little", signed=True)


def packed_text(xy: np.ndarray, end: np.ndarray) -> str:
    """The Kaggle "simplified" JSON of a packed drawing, without spaces."""
    parts = []
    for stroke in np.split(np.asarray(xy, dtype=np.int64), np.flatnonzero(end)[:-1] + 1):
        parts.append("[[" + ",".join(map(str, stroke[:, 0])) + "],[" + ",".join(map(str, stroke[:, 1])) + "]]")
    return "[" + ",".join(parts) + "]"


def pack(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Points and pen lifts of one drawing, as in quickdraw.prepare."""
    strokes = json.loads(text)
    p = np.array([(x, y) for xs, ys in strokes for x, y in zip(xs, ys)], dtype=np.int64)
    if len(p) == 0 or p.min() < 0 or p.max() > 255:
        raise ValueError("coordinates outside 0..255")
    e = np.zeros(len(p), dtype=bool)
    e[np.cumsum([len(xs) for xs, _ in strokes]) - 1] = True
    return p.astype(np.uint8), e


def used_drawings(root: Path) -> tuple[set, set, dict]:
    """key_ids and stroke hashes of every split v1 drawing, and word -> label."""
    keys, hashes, labels = set(), set(), {}
    manifest = json.loads((root / "manifest.json").read_text())
    keys.update(int(k) for k in manifest.get("dropped_key_ids", []))
    for split in USED_SPLITS:
        meta = pd.read_parquet(root / f"{split}_meta.parquet", columns=["key_id", "label", "word"])
        keys.update(meta["key_id"].astype(np.int64).tolist())
        labels.update(dict(zip(meta["word"], meta["label"].astype(int))))
        offsets = np.load(root / f"{split}_offsets.npy")
        xy = np.load(root / f"{split}_xy.npy", mmap_mode="r")
        end = np.load(root / f"{split}_end.npy", mmap_mode="r")
        for i in range(len(offsets) - 1):
            a, b = offsets[i], offsets[i + 1]
            hashes.add(text_hash(packed_text(xy[a:b], end[a:b])))
        print(f"split v1 {split}: {len(meta):,} drawings", flush=True)
    return keys, hashes, labels


def read_class(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["countrycode", "drawing", "key_id", "recognized", "word"],
                     dtype={"countrycode": str, "drawing": str, "word": str}, keep_default_na=False,
                     na_values={"countrycode": [""]})
    df["key_id"] = df["key_id"].astype(np.int64)
    df["recognized"] = df["recognized"].astype(str).str.lower().eq("true")
    return df


def build(kaggle: Path, root: Path, out: Path, spec: dict) -> dict:
    folder = kaggle / "train_simplified"
    holdout = set(json.loads(Path(spec["holdout_file"]).read_text())["countries"])
    keys, v1_hashes, labels = used_drawings(root)
    words = sorted(labels, key=labels.get)
    if len(words) != 100 or [labels[w] for w in words] != list(range(100)):
        raise ValueError("split v1 labels are not 0..99")

    per_class, reserve, seed = spec["per_class"], spec["reserve"], spec["seed"]
    v1_array = np.fromiter(v1_hashes, np.int64, count=len(v1_hashes))
    counts, pools, all_hashes, all_labels = {}, {}, [], []
    for word in words:
        path = folder / f"{word}.csv"
        df = read_class(path)
        if (df["word"] != word).any():
            raise ValueError(f"{path.name}: rows of another class")
        h = np.array([text_hash(s) for s in df["drawing"]], dtype=np.int64)
        c = {"rows": int(len(df)), "used_key_id": int(df["key_id"].isin(keys).sum())}
        keep = ~df["key_id"].isin(keys).to_numpy()
        held = df["countrycode"].isin(holdout).to_numpy()
        c["held_out_country"] = int((keep & held).sum())
        keep &= ~held
        dup_v1 = np.isin(h, v1_array)
        c["repeats_split_v1"] = int((keep & dup_v1).sum())
        keep &= ~dup_v1
        # First of identical copies within the class (smallest key_id).
        order = np.argsort(df["key_id"].to_numpy(), kind="stable")
        first = np.zeros(len(df), bool)
        _, idx = np.unique(h[order], return_index=True)
        first[order[idx]] = True
        c["repeat_within_class"] = int((keep & ~first).sum())
        keep &= first
        all_hashes.append(h[keep])
        all_labels.append(np.full(int(keep.sum()), labels[word]))
        pos = np.flatnonzero(keep)
        rng = np.random.default_rng([seed, labels[word]])
        pick = pos[rng.permutation(len(pos))[:per_class + reserve]]
        pools[word] = df.iloc[pick].assign(hash=h[pick]).reset_index(drop=True)
        c["eligible_before_conflicts"] = int(len(pos))
        counts[word] = c
        print(f"{word}: {len(df):,} rows, {len(pos):,} eligible", flush=True)

    # Identical strokes under more than one class among the candidates: drop them all.
    h_all, l_all = np.concatenate(all_hashes), np.concatenate(all_labels)
    pairs = np.unique(np.stack([h_all, l_all]), axis=1)
    hashes, n_classes = np.unique(pairs[0], return_counts=True)
    conflict = set(hashes[n_classes > 1].tolist())

    metas, xys, ends = [], [], []
    for word in words:
        pool = pools[word]
        bad = pool["hash"].isin(conflict).to_numpy()
        counts[word]["label_conflict_in_reserve"] = int(bad.sum())
        chosen = pool[~bad].head(per_class)
        if len(chosen) < per_class:
            raise ValueError(f"{word}: only {len(chosen)} drawings left")
        for text in chosen["drawing"]:
            xy, end = pack(text)
            xys.append(xy)
            ends.append(end)
        stats = [_stats(x, e) for x, e in zip(xys[-per_class:], ends[-per_class:])]
        m = chosen[["key_id", "word", "recognized", "countrycode"]].copy()
        m["label"] = labels[word]
        m["n_points"] = [s[0] for s in stats]
        m["n_strokes"] = [s[1] for s in stats]
        metas.append(m)
    meta = pd.concat(metas, ignore_index=True)
    meta.insert(1, "row", np.arange(len(meta)))
    meta = meta[["key_id", "row", "label", "word", "recognized", "countrycode", "n_points", "n_strokes"]]

    out.mkdir(parents=True, exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum([len(x) for x in xys])]).astype(np.int64)
    np.save(out / f"{SPLIT}_xy.npy", np.concatenate(xys))
    np.save(out / f"{SPLIT}_end.npy", np.concatenate(ends))
    np.save(out / f"{SPLIT}_offsets.npy", offsets)
    meta.to_parquet(out / f"{SPLIT}_meta.parquet", index=False)
    if meta["key_id"].isin(keys).any() or meta["key_id"].duplicated().any():
        raise RuntimeError("a used or repeated key_id reached test_v2")

    totals = {k: int(sum(c[k] for c in counts.values())) for k in next(iter(counts.values()))}
    manifest = {"split": SPLIT, "config": spec, "drawings": int(len(meta)), "per_class": per_class,
                "totals": totals, "per_class_counts": counts,
                "recognized_share": float(meta["recognized"].mean()),
                "countries": int(meta["countrycode"].nunique()),
                "us_share": float((meta["countrycode"] == "US").mean()),
                "key_id_digest": hashlib.sha256("\n".join(map(str, sorted(meta["key_id"]))).encode()).hexdigest(),
                "files": {p.name: _sha256(p) for p in sorted(out.glob(f"{SPLIT}_*"))}}
    (out / f"manifest_{SPLIT}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def _stats(xy: np.ndarray, end: np.ndarray) -> tuple[int, int]:
    return int(len(xy)), int(end.sum())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle", required=True, help="the competition folder, holding train_simplified/")
    parser.add_argument("--data-root", default=None, help="processed split v1 (default: found automatically)")
    parser.add_argument("--config", default="configs/test_v2.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from .config import repo_path
    from .paths import resolve_data_root
    spec = yaml.safe_load(repo_path(args.config).read_text(encoding="utf-8"))
    spec["holdout_file"] = str(repo_path(spec["holdout_file"]))
    m = build(Path(args.kaggle), resolve_data_root(args.data_root), Path(args.out), spec)
    print(json.dumps({k: m[k] for k in ("drawings", "totals", "recognized_share", "countries", "us_share")}, indent=1))
    print(f"\nTEST_V2 BUILT. Records: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
