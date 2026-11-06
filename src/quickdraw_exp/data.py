"""Data access on top of ``quickdraw.dataset``, with the country hold-out.

Split v1 and its files stay unchanged: held-out countries are filtered at load
time. The test data can only be opened by asking for it explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_holdout(path) -> list[str]:
    countries = json.loads(Path(path).read_text(encoding="utf-8"))["countries"]
    if not countries or len(set(countries)) != len(countries):
        raise ValueError(f"bad hold-out list in {path}")
    return countries


class SubsetSplit:
    """The drawings of a split at ``indices``; usable wherever a
    ``QuickDrawSplit`` is. Randomness in the views depends on the position here."""

    def __init__(self, base, indices):
        self.base, self.indices = base, np.asarray(indices, dtype=np.int64)
        self.split, self.root = base.split, base.root
        self.meta = base.meta.iloc[self.indices].reset_index(drop=True)
        self.labels = np.asarray(base.labels)[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[int(self.indices[i])]


class ConcatSplit:
    """Several splits read as one, for the unseen-country test."""

    def __init__(self, parts, name: str):
        self.parts, self.split, self.root = parts, name, parts[0].root
        self.meta = pd.concat([p.meta for p in parts], ignore_index=True)
        self.labels = np.concatenate([np.asarray(p.labels) for p in parts])
        self._starts = np.cumsum([0] + [len(p) for p in parts])

    def __len__(self) -> int:
        return int(self._starts[-1])

    def __getitem__(self, i: int):
        part = int(np.searchsorted(self._starts, i, side="right")) - 1
        return self.parts[part][int(i - self._starts[part])]


def country_mask(meta: pd.DataFrame, countries) -> np.ndarray:
    """True for drawings of ``countries``. A missing country code is False."""
    return meta["countrycode"].isin(list(countries)).to_numpy()


def without_countries(base, countries) -> SubsetSplit:
    return SubsetSplit(base, np.flatnonzero(~country_mask(base.meta, countries)))


def only_countries(base, countries) -> SubsetSplit:
    return SubsetSplit(base, np.flatnonzero(country_mask(base.meta, countries)))


def open_split(split: str, root, holdout, final_evaluation: bool = False):
    """Training, validation or test data without the held-out countries."""
    from quickdraw.dataset import QuickDrawSplit
    if split == "test" and not final_evaluation:
        raise PermissionError("the test split is only for the final evaluation")
    return without_countries(QuickDrawSplit(split, root=Path(root)), holdout)


def open_unseen_countries(root, holdout, final_evaluation: bool = False) -> ConcatSplit:
    """Every drawing of the held-out countries, across the three v1 splits."""
    from quickdraw.dataset import QuickDrawSplit
    if not final_evaluation:
        raise PermissionError("the unseen-country test is only for the final evaluation")
    parts = [only_countries(QuickDrawSplit(s, root=Path(root)), holdout) for s in ("train", "val", "test")]
    return ConcatSplit(parts, "unseen_countries")


class ResumableSampler:
    """A shuffled order fixed by (seed, epoch) that can start part-way through,
    so training restarted mid-epoch continues with the very next batch."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set_position(self, epoch: int, start: int = 0) -> None:
        if not 0 <= start <= self.n:
            raise ValueError(f"start {start} outside 0..{self.n}")
        self.epoch, self.start = epoch, start

    def __iter__(self):
        order = np.random.default_rng([self.seed, self.epoch]).permutation(self.n)
        return iter(order[self.start:].tolist())

    def __len__(self) -> int:
        return self.n - self.start
