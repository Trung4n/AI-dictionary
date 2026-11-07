"""Fit the delta scale of resampled sequences on the training split only.

    .venv/Scripts/python scripts/fit_resampled_stats.py

Uses random prefixes, as ``SequenceView`` does in training (a fixed seed and a
sample of the split are enough for a scale). Writes ``src/quickdraw/seq_stats_resampled.json``.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from quickdraw.dataset import RESAMPLED_STATS_PATH, QuickDrawSplit
from quickdraw.transforms import RESAMPLE_STEP, normalize, prefix, resample_uniform

N = 200_000
SEED = 42
FRAC_RANGE = (0.05, 1.0)


def main():
    ds = QuickDrawSplit("train")
    rng = np.random.default_rng(SEED)
    s, s2, count = np.zeros(2), np.zeros(2), 0
    for i in rng.choice(len(ds), N, replace=False):
        pts, end, _ = ds[int(i)]
        p, e = prefix(pts, end, rng.uniform(*FRAC_RANGE))
        r, _ = resample_uniform(normalize(p), e, RESAMPLE_STEP)
        d = np.diff(r.astype(np.float64), axis=0)
        s += d.sum(axis=0)
        s2 += (d ** 2).sum(axis=0)
        count += len(d)
    mean = s / count
    std = np.sqrt(s2 / count - mean ** 2)
    stats = {"fitted_on": "train", "step": RESAMPLE_STEP, "n_drawings": N, "seed": SEED,
             "frac_range": list(FRAC_RANGE), "n_deltas": count,
             "delta_mean": mean.round(6).tolist(), "delta_std": std.round(6).tolist()}
    RESAMPLED_STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(stats)


if __name__ == "__main__":
    main()
