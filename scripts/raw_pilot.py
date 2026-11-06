"""Pilot on raw QuickDraw strokes for a few classes.

    .venv/Scripts/python -I scripts/raw_pilot.py <raw_dir> triangle bat

<raw_dir> holds ``{class}.ndjson`` from gs://quickdraw_dataset/full/raw/. Answers:
1. Do raw ``key_id``s join with our subset, and does our simplify match the stored data?
2. How long do drawings last, by ``recognized``? (the stopping mechanism)
3. Does cutting the raw strokes first and simplifying only the prefix remove the scale
   leak? Same drawings and fractions in every arm.

Rows of our test split are excluded everywhere. Writes
``reports/preprocessing/raw_pilot.json`` and a figure.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quickdraw.dataset import QuickDrawSplit
from quickdraw.leak import (attack_all_points, attack_stroke_endpoints, cnn_r2, cv_r2,
                            hit_rate, seq_features, side)
from quickdraw.prepare import REPORT_DIR
from quickdraw.simplify import simplify
from quickdraw.transforms import (RESAMPLE_STEP, ink_lengths, normalize, prefix, rasterize,
                                  resample_uniform)

N_PER_CLASS = 20_000
N_VALIDATE = 2_000
SEED = 0
FRAC_RANGE = (0.05, 1.0)
SUBSET = (0.2, 0.3)
TIME_LIMIT_MS = 20_000


def pack_raw(drawing):
    pts = np.array([(x, y) for s in drawing for x, y in zip(s[0], s[1])], dtype=np.float64)
    t = np.array([v for s in drawing for v in s[2]], dtype=np.float64)
    end = np.zeros(len(pts), dtype=bool)
    end[np.cumsum([len(s[0]) for s in drawing]) - 1] = True
    return pts, end, t


def chamfer(a, b):
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    return (d.min(axis=1).mean() + d.min(axis=0).mean()) / 2


def main(raw_dir, classes):
    rng = np.random.default_rng(SEED)
    splits = {s: QuickDrawSplit(s) for s in ("train", "val")}
    test_meta = pd.read_parquet(splits["val"].root / "test_meta.parquet", columns=["key_id", "word"])
    result = {"classes": classes, "seed": SEED, "per_class": {}, "excluded": "our test split"}
    arms = {k: [] for k in ("stored", "stored_resampled", "clean", "clean_resampled")}
    ratio_stored, ratio_clean, fracs_all, time_fracs, labels = [], [], [], [], []
    images = {"stored": [], "clean": []}
    attacks = {f"{a}_{k}": [] for a in ("stored", "stored_resampled", "clean", "clean_resampled")
               for k in ("all", "endpoints")}
    durations = {}

    for label, word in enumerate(classes):
        meta = {s: d.meta[d.meta["word"] == word] for s, d in splits.items()}
        ours = pd.concat(meta.values())
        test_keys = set(test_meta.loc[test_meta["word"] == word, "key_id"])
        pick = ours.sample(min(N_PER_CLASS, len(ours)), random_state=SEED)["key_id"]
        chosen, raw_rec, rows = set(pick), {}, []
        with open(Path(raw_dir) / f"{word}.ndjson", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                key = int(obj["key_id"])
                if key in test_keys:
                    continue
                d = obj["drawing"]
                rows.append((key, bool(obj["recognized"]), max(s[2][-1] for s in d), len(d)))
                if key in chosen:
                    raw_rec[key] = d
        raw = pd.DataFrame(rows, columns=["key_id", "recognized", "duration_ms", "n_strokes"])
        del rows
        durations[word] = raw[["recognized", "duration_ms"]]

        joined = ours[["key_id", "recognized"]].merge(raw, on="key_id", how="left", suffixes=("", "_raw"))
        q = raw.groupby("recognized")["duration_ms"].quantile([0.5, 0.9, 0.99]).unstack()
        at_limit = raw.assign(at=raw["duration_ms"] >= TIME_LIMIT_MS - 1000).groupby("recognized")["at"].mean()
        over = raw.assign(o=raw["duration_ms"] > TIME_LIMIT_MS + 1000).groupby("recognized")["o"].mean()
        info = {
            "raw_rows_non_test": int(len(raw)),
            "our_rows_train_val": int(len(ours)),
            "our_rows_found_in_raw": float(joined["duration_ms"].notna().mean()),
            "recognized_flag_agrees": float((joined["recognized"] == joined["recognized_raw"]).mean()),
            "recognized_rate_raw": float(raw["recognized"].mean()),
            "duration_ms_quantiles": {str(k): {str(c): float(v) for c, v in r.items()} for k, r in q.iterrows()},
            "share_duration_within_1s_of_limit_or_more": {str(k): float(v) for k, v in at_limit.items()},
            "share_duration_over_limit_plus_1s": {str(k): float(v) for k, v in over.items()},
        }

        by_key = {}
        for s, m in meta.items():
            sel = m[m["key_id"].isin(raw_rec)]
            for i, key in zip(sel.index, sel["key_id"]):
                by_key[int(key)] = (s, int(i))
        keys = [k for k in pick if k in raw_rec and k in by_key]
        validate, exact, pts_ratio = [], 0, []
        for j, key in enumerate(keys):
            s, i = by_key[key]
            pts, end, _ = splits[s][i]
            rp, re, rt = pack_raw(raw_rec[key])
            if j < N_VALIDATE:
                sp, se = simplify(rp, re)
                validate.append(chamfer(sp, pts.astype(np.float64)))
                pts_ratio.append(len(sp) / len(pts))
                exact += sp.shape == pts.shape and bool((sp == pts).all())
            frac = rng.uniform(*FRAC_RANGE)
            p, e = prefix(pts, end, frac)
            rq, rqe = prefix(rp, re, frac)
            cq, cqe = simplify(rq, rqe)
            ratio_stored.append(max(side(p), 1e-3) / max(side(pts), 1.0))
            ratio_clean.append(max(side(rq), 1e-3) / max(side(rp), 1.0))
            fracs_all.append(frac)
            labels.append(label)
            # Where in time the ink cut falls.
            ink = np.concatenate([[0.0], np.cumsum(ink_lengths(rp, re))])
            total_t = rt[-1] if rt[-1] > 0 else 1.0
            time_fracs.append(rt[min(int(np.searchsorted(ink, frac * ink[-1])), len(rt) - 1)] / total_t)

            n = normalize(p)
            c = normalize(cq)
            views = {"stored": (n, e), "clean": (c, cqe),
                     "stored_resampled": resample_uniform(n, e, RESAMPLE_STEP),
                     "clean_resampled": resample_uniform(c, cqe, RESAMPLE_STEP)}
            for a, (vp, ve) in views.items():
                arms[a].append(seq_features(vp, ve))
                attacks[f"{a}_all"].append(attack_all_points(vp))
                attacks[f"{a}_endpoints"].append(attack_stroke_endpoints(vp, ve))
            images["stored"].append(rasterize(n, e, 64))
            images["clean"].append(rasterize(c, cqe, 64))
        info["n_arm_drawings"] = len(keys)
        info["our_simplify_vs_stored"] = {
            "n": min(len(keys), N_VALIDATE), "exact_match": exact / max(min(len(keys), N_VALIDATE), 1),
            "chamfer_median_canvas_units": float(np.median(validate)),
            "chamfer_p99": float(np.quantile(validate, 0.99)),
            "point_count_ratio_median": float(np.median(pts_ratio)),
        }
        result["per_class"][word] = info
        print(word, json.dumps(info, indent=1), flush=True)
        del raw_rec

    fracs_all = np.array(fracs_all)
    ratio_stored, ratio_clean = np.array(ratio_stored), np.array(ratio_clean)
    labels = np.array(labels)
    time_fracs = np.array(time_fracs)
    subset = (fracs_all >= SUBSET[0]) & (fracs_all <= SUBSET[1])
    result["ink_frac_vs_time_frac"] = {
        f"{a:.2f}-{b:.2f}": float(np.median(time_fracs[(fracs_all >= a) & (fracs_all < b)]))
        for a, b in [(0.05, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]}
    result["ink_time_corr"] = float(np.corrcoef(fracs_all, time_fracs)[0, 1])

    target = {a: (ratio_clean if a.startswith("clean") else ratio_stored) for a in arms}
    result["base_rate_ratio_about_1"] = {"stored": hit_rate(np.ones(len(ratio_stored)), ratio_stored),
                                         "clean": hit_rate(np.ones(len(ratio_clean)), ratio_clean)}
    result["grid_attack_hit_rate"] = {k: hit_rate(v, target[k.rsplit("_", 1)[0]]) for k, v in attacks.items()}
    result["gb_r2"] = {}
    for a, feats in arms.items():
        X = np.asarray(feats, np.float32)
        y = np.log(target[a])
        result["gb_r2"][a] = {"all": cv_r2(X, y, subset),
                              "by_class": {w: cv_r2(X[labels == k], y[labels == k]) for k, w in enumerate(classes)}}
        print(a, result["gb_r2"][a], flush=True)
    result["cnn_raster64_r2"] = {}
    for a in ("stored", "clean"):
        result["cnn_raster64_r2"][a] = cnn_r2(np.stack(images[a]), np.log(target[a]), subset)
        print("cnn", a, result["cnn_raster64_r2"][a], flush=True)

    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 3), squeeze=False)
    for ax, word in zip(axes[0], classes):
        d = durations[word]
        for flag, name in ((True, "recognized"), (False, "not recognized")):
            ax.hist(np.minimum(d.loc[d["recognized"] == flag, "duration_ms"], 30_000) / 1000, bins=60,
                    density=True, alpha=0.6, label=name)
        ax.axvline(TIME_LIMIT_MS / 1000, color="k", ls=":", lw=1)
        ax.set(title=word, xlabel="time of last point (s, clipped at 30)")
        ax.legend()
    fig.savefig(REPORT_DIR / "figures" / "raw_duration_by_recognized.png", dpi=150, bbox_inches="tight")
    (REPORT_DIR / "raw_pilot.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("ink_frac_vs_time_frac", "ink_time_corr", "base_rate_ratio_about_1",
                                             "grid_attack_hit_rate")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
