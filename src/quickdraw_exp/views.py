"""Input normalisations compared in the leak experiment (step 3).

``prefix``     the preprocessing pipeline as it stands: each prefix is scaled by
               its own bounding box. Coordinates stay on a lattice set by the
               prefix's size, and vertices were simplified at the full drawing's
               scale, so traces of that scale may remain.
``full_bbox``  the prefix keeps the full drawing's frame, as when a sequence model
               is read out step by step over coordinates normalised to the whole
               drawing. Impossible while drawing, because the final size is unknown.
``resampled``  ``prefix``, then each stroke is resampled at even spacing along its
               length, which removes the lattice and the vertex density.
"""
from __future__ import annotations

import numpy as np
from quickdraw.dataset import RasterView, SequenceView
from quickdraw.transforms import CANVAS, augment, normalize, prefix, stroke_ids

INPUT_NORMS = ("prefix", "full_bbox", "resampled")


def resample_strokes(pts: np.ndarray, end: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``spacing`` units along each stroke, keeping both its ends."""
    pts = np.asarray(pts, dtype=np.float64)
    out_pts, out_end = [], []
    ids = stroke_ids(end)
    for s in range(int(ids[-1]) + 1):
        stroke = pts[ids == s]
        if len(stroke) < 2:
            out_pts.append(stroke)
            out_end.append(np.ones(len(stroke), dtype=bool))
            continue
        along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(stroke, axis=0).T))])
        n = max(2, int(round(along[-1] / spacing)) + 1)
        t = np.linspace(0.0, along[-1], n)
        out_pts.append(np.column_stack([np.interp(t, along, stroke[:, 0]), np.interp(t, along, stroke[:, 1])]))
        flags = np.zeros(n, dtype=bool)
        flags[-1] = True
        out_end.append(flags)
    return np.concatenate(out_pts).astype(np.float32), np.concatenate(out_end)


class _NormMixin:
    """Replaces the view's prefix, augment and normalise step. Randomness comes
    from the same generator in the same order, so at one (seed, epoch, index) every
    condition sees the same fraction and the same random transform."""

    def _configure_norm(self, input_norm: str, spacing: float) -> None:
        if input_norm not in INPUT_NORMS:
            raise ValueError(f"unknown input_norm {input_norm!r}")
        self.input_norm, self.spacing = input_norm, spacing

    def _prepare(self, i: int):
        if self.input_norm == "prefix":
            return super()._prepare(i)
        if self.input_norm == "resampled":
            pts, end, label = super()._prepare(i)
            pts, end = resample_strokes(pts, end, self.spacing)
            return normalize(pts), end, label
        return self._prepare_full_bbox(i)

    def _prepare_full_bbox(self, i: int):
        pts, end, label = self.base[i]
        rng = np.random.default_rng([self.seed, self.epoch, i])
        frac = rng.uniform(*self.frac_range) if self.frac == "random" else self.frac
        part, part_end = pts, end
        if frac is not None and frac < 1:
            part, part_end = prefix(pts, end, frac, self.prefix_mode)
        k = len(part)
        both = np.vstack([np.asarray(part, dtype=np.float32), np.asarray(pts, dtype=np.float32)])
        if self.aug is not None:
            both = augment(both, rng, self.aug)
        whole = both[k:]
        lo = whole.min(axis=0)
        side = float((whole.max(axis=0) - lo).max())
        return (both[:k] - lo) * (CANVAS / side if side > 0 else 1.0), part_end, label


class NormRasterView(_NormMixin, RasterView):
    def __init__(self, base, input_norm: str = "prefix", spacing: float = 16.0, **kw):
        super().__init__(base, **kw)
        self._configure_norm(input_norm, spacing)


class NormSequenceView(_NormMixin, SequenceView):
    def __init__(self, base, input_norm: str = "prefix", spacing: float = 16.0, **kw):
        super().__init__(base, **kw)
        self._configure_norm(input_norm, spacing)


class MultiPrefixView:
    """``k`` prefixes of the same drawing, in drawing order, for training.

    The fractions are drawn uniformly from ``frac_range`` and sorted. Both prefixes
    get the same random affine transform, so they differ only in how much is drawn.
    Returns ``(images[k, size, size], label)`` or ``(seqs[k, max_len, 3], lengths[k], label)``.
    """

    def __init__(self, view, k: int):
        if view.frac != "random":
            raise ValueError("MultiPrefixView needs a training view with random prefixes")
        self.view, self.k = view, k
        self.split, self.root = view.base.split, view.base.root

    def set_epoch(self, epoch: int) -> None:
        self.view.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.view)

    def __getitem__(self, i: int):
        view = self.view
        rng = np.random.default_rng([view.seed, view.epoch, i, 1])
        fracs = np.sort(rng.uniform(*view.frac_range, size=self.k))
        outs = []
        try:
            for frac in fracs:
                view.frac = float(frac)
                outs.append(view[i])
        finally:
            view.frac = "random"
        label = outs[0][-1]
        if len(outs[0]) == 2:
            return np.stack([o[0] for o in outs]), label
        return np.stack([o[0] for o in outs]), np.array([o[1] for o in outs], dtype=np.int64), label
