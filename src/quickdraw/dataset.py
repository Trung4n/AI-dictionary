"""Reading the packed splits written by ``prepare.py``.

The classes here do not import torch but follow the map-style dataset protocol
(``__len__`` / ``__getitem__``), so they can be passed straight to
``torch.utils.data.DataLoader``. Arrays are memory-mapped: worker processes share
the page cache instead of each holding a copy.

    train = RasterView(QuickDrawSplit("train"), frac="random", augment=AugmentConfig())
    val25 = RasterView(QuickDrawSplit("val"), frac=0.25)      # deterministic
    train.set_epoch(epoch)                                     # new randomness each epoch
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare import PROCESSED_DIR
from .transforms import (AugmentConfig, augment, normalize, prefix, rasterize,
                         to_sequence)


class QuickDrawSplit:
    """Raw drawings of one split: ``ds[i] -> (pts uint8 (P, 2), end bool (P,), label)``."""

    def __init__(self, split: str, root: Path = PROCESSED_DIR):
        self.split, self.root = split, root
        self.meta = pd.read_parquet(root / f"{split}_meta.parquet")
        self.labels = self.meta["label"].to_numpy()
        self.offsets = np.load(root / f"{split}_offsets.npy")
        self.xy = np.load(root / f"{split}_xy.npy", mmap_mode="r")
        self.end = np.load(root / f"{split}_end.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.labels)

    def drawing(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.offsets[i], self.offsets[i + 1]
        return np.asarray(self.xy[a:b]), np.asarray(self.end[a:b])

    def __getitem__(self, i: int):
        pts, end = self.drawing(i)
        return pts, end, int(self.labels[i])


def load_seq_stats(root: Path = PROCESSED_DIR) -> tuple[float, float]:
    return tuple(json.loads((root / "seq_stats.json").read_text())["delta_std"])


class _View:
    """Shared prefix/augment/normalise logic.

    ``frac``: ``None`` or ``1.0`` = full drawing; a float = that fixed fraction
    (evaluation); ``"random"`` = uniform in ``frac_range`` per sample (training).
    Randomness comes from ``rng([seed, epoch, index])``, so a sample is
    reproducible, differs between epochs, and is independent of worker count.
    """

    def __init__(self, base: QuickDrawSplit, frac=None, prefix_mode: str = "ink",
                 augment: AugmentConfig | None = None, frac_range=(0.05, 1.0), seed: int = 0):
        if frac == "random" and base.split != "train":
            raise ValueError("random prefixes are for training; evaluate at fixed fractions")
        self.base, self.frac, self.prefix_mode = base, frac, prefix_mode
        self.aug, self.frac_range, self.seed, self.epoch = augment, frac_range, seed, 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.base)

    def _prepare(self, i: int):
        pts, end, label = self.base[i]
        rng = np.random.default_rng([self.seed, self.epoch, i])
        frac = rng.uniform(*self.frac_range) if self.frac == "random" else self.frac
        if frac is not None and frac < 1:
            pts, end = prefix(pts, end, frac, self.prefix_mode)
        if self.aug is not None:
            pts = augment(pts, rng, self.aug)
        return normalize(pts), end, label


class RasterView(_View):
    """``view[i] -> (image uint8 [size, size], label)``."""

    def __init__(self, base, size: int = 64, order_encoding: bool = False, **kw):
        super().__init__(base, **kw)
        self.size, self.order_encoding = size, order_encoding

    def __getitem__(self, i: int):
        pts, end, label = self._prepare(i)
        return rasterize(pts, end, self.size, order_encoding=self.order_encoding), label


class SequenceView(_View):
    """``view[i] -> (seq float32 [max_len, 3], length, label)``."""

    def __init__(self, base, max_len: int = 256, delta_std=None, **kw):
        super().__init__(base, **kw)
        self.max_len = max_len
        self.delta_std = delta_std or load_seq_stats(base.root)

    def __getitem__(self, i: int):
        pts, end, label = self._prepare(i)
        seq, n = to_sequence(pts, end, self.max_len, self.delta_std)
        return seq, n, label
