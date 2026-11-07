"""Scale-leak audit of the current pipeline on 100 classes (validation split only).

    .venv/Scripts/python scripts/leak_audit.py

Writes ``reports/preprocessing/leak_audit.json``. The test split is never read.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from quickdraw.dataset import QuickDrawSplit
from quickdraw.leak import (attack_all_points, attack_stroke_endpoints, cnn_r2, cv_r2,
                            hit_rate, seq_features, side)
from quickdraw.prepare import REPORT_DIR
from quickdraw.transforms import RESAMPLE_STEP, normalize, prefix, rasterize, resample_uniform

N = 40_000
SEED = 0
FRAC_RANGE = (0.05, 1.0)
SUBSET = (0.2, 0.3)


def main():
    ds = QuickDrawSplit("val")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(ds), N, replace=False)
    fracs = rng.uniform(*FRAC_RANGE, N)
    ratio = np.zeros(N)
    attack = {k: [] for k in ("stored_all", "stored_endpoints", "resampled_all", "resampled_endpoints")}
    f_stored, f_res = [], []
    images = np.zeros((N, 64, 64), dtype=np.uint8)
    iou = []
    for j, i in enumerate(idx):
        pts, end, _ = ds[int(i)]
        p, e = prefix(pts, end, fracs[j])
        ratio[j] = max(side(p), 1e-3) / max(side(pts), 1.0)
        n = normalize(p)
        r, re = resample_uniform(n, e, RESAMPLE_STEP)
        attack["stored_all"].append(attack_all_points(n))
        attack["stored_endpoints"].append(attack_stroke_endpoints(n, e))
        attack["resampled_all"].append(attack_all_points(r))
        attack["resampled_endpoints"].append(attack_stroke_endpoints(r, re))
        f_stored.append(seq_features(n, e))
        f_res.append(seq_features(r, re))
        images[j] = rasterize(n, e, 64)
        a, b = images[j] > 127, rasterize(normalize(r), re, 64) > 127
        iou.append((a & b).sum() / max((a | b).sum(), 1))
    y = np.log(ratio)
    subset = (fracs >= SUBSET[0]) & (fracs <= SUBSET[1])
    result = {
        "n": N, "split": "val", "seed": SEED, "frac_range": FRAC_RANGE, "subset_frac": SUBSET,
        "resample_step": RESAMPLE_STEP,
        "base_rate_ratio_about_1": hit_rate(np.ones(N), ratio),
        "grid_attack_hit_rate": {k: hit_rate(v, ratio) for k, v in attack.items()},
        "grid_attack_hit_rate_ratio_below_0.9": {
            k: hit_rate(np.asarray(v)[ratio < 0.9], ratio[ratio < 0.9]) for k, v in attack.items()},
        "share_ratio_below_0.9": float((ratio < 0.9).mean()),
        "resample_raster_iou_median": float(np.median(iou)),
        "resample_raster_iou_p10": float(np.quantile(iou, 0.1)),
        "gb_seq_stored": cv_r2(np.asarray(f_stored, np.float32), y, subset),
        "gb_seq_resampled": cv_r2(np.asarray(f_res, np.float32), y, subset),
    }
    print(json.dumps(result, indent=2), flush=True)
    result["cnn_raster64_stored"] = cnn_r2(images, y, subset)
    print(result["cnn_raster64_stored"], flush=True)
    (REPORT_DIR / "leak_audit.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
