"""Per-sample transforms on the packed representation.

A drawing is ``(pts, end)``: ``pts`` is a ``(P, 2)`` array of x, y coordinates
in drawing order and ``end`` a ``(P,)`` bool array that is True on the last point
of every stroke. This is what ``dataset.QuickDrawSplit`` returns.

Order used by the views in ``dataset.py`` (decisions from notebooks/01_eda.ipynb):

    prefix (on raw coordinates) -> augment -> normalize -> rasterize / to_sequence

Normalising *after* taking the prefix means a prefix is scaled by its own
bounding box, so its size does not reveal how large the finished drawing is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

CANVAS = 255
EVAL_FRACS = (0.1, 0.25, 0.5, 0.75, 1.0)
PREFIX_MODES = ("ink", "points", "strokes_ceil")
RESAMPLE_STEP = 8.0  # canvas units; about 2 px of a 64 x 64 raster


def stroke_ids(end: np.ndarray) -> np.ndarray:
    """Index of the stroke each point belongs to."""
    return np.concatenate([[0], np.cumsum(end[:-1])]).astype(np.int64)


def ink_lengths(pts: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Length of the segment from point i to i+1; 0 across a pen lift."""
    d = np.hypot(*np.diff(pts.astype(np.float64), axis=0).T)
    d[end[:-1]] = 0.0
    return d


# ---------------------------------------------------------------- prefix

def prefix(pts: np.ndarray, end: np.ndarray, frac: float, mode: str = "ink"):
    """The first ``frac`` of a drawing, never empty.

    * ``ink``: points up to ``frac`` of the total ink length; the last segment is
      interpolated. Drawings made only of dots (zero ink) fall back to ``points``.
    * ``points``: the first ``ceil(frac * P)`` points (at least 1).
    * ``strokes_ceil``: the first ``ceil(frac * n_strokes)`` whole strokes (at
      least 1); the definition the class-selection script appears to use.
    """
    if frac >= 1:
        return pts, end
    if not frac > 0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if mode == "strokes_ceil":
        k = max(1, math.ceil(frac * (int(end.sum()))))
        n = int(np.flatnonzero(end)[k - 1]) + 1
        return pts[:n], end[:n]
    if mode == "ink":
        d = ink_lengths(pts, end)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        if cum[-1] > 0:
            target = frac * cum[-1]
            k = int(np.searchsorted(cum, target, side="left"))  # cum[k-1] < target <= cum[k]
            t = (target - cum[k - 1]) / d[k - 1]
            p0, p1 = pts[k - 1].astype(np.float64), pts[k].astype(np.float64)
            out_pts = np.vstack([pts[:k].astype(np.float64), p0 + t * (p1 - p0)])
            out_end = np.concatenate([end[:k], [True]])
            return out_pts, out_end
        mode = "points"
    if mode == "points":
        n = max(1, math.ceil(frac * len(pts) - 1e-9))
        out_end = end[:n].copy()
        out_end[-1] = True
        return pts[:n], out_end
    raise ValueError(f"unknown prefix mode {mode!r}")


# ---------------------------------------------------------------- geometry

def normalize(pts: np.ndarray) -> np.ndarray:
    """Shift to the top-left corner and scale the longer side to ``CANVAS``,
    keeping the aspect ratio (Kaggle "simplified" convention). A drawing with
    no extent (a single dot) is only shifted."""
    pts = pts.astype(np.float32)
    lo = pts.min(axis=0)
    side = float((pts.max(axis=0) - lo).max())
    return (pts - lo) * (CANVAS / side if side > 0 else 1.0)


@dataclass(frozen=True)
class AugmentConfig:
    """Random affine applied to raw coordinates before normalisation. Flips are
    off by default: the EDA did not check whether every class is symmetric."""
    rotate_deg: float = 10.0
    scale: tuple[float, float] = (0.9, 1.1)    # per axis, so aspect ratio jitters too
    shear: float = 0.1
    jitter_px: float = 0.0                     # Gaussian noise per point, in raw pixels
    hflip_p: float = 0.0


def augment(pts: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig = AugmentConfig()) -> np.ndarray:
    pts = pts.astype(np.float32)
    a = math.radians(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
    sx, sy = rng.uniform(*cfg.scale, size=2)
    sh = rng.uniform(-cfg.shear, cfg.shear)
    if rng.random() < cfg.hflip_p:
        sx = -sx
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=np.float32)
    m = rot @ np.array([[1, sh], [0, 1]], dtype=np.float32) @ np.diag([sx, sy]).astype(np.float32)
    c = (pts.min(axis=0) + pts.max(axis=0)) / 2
    out = (pts - c) @ m.T + c
    if cfg.jitter_px > 0:
        out += rng.normal(0, cfg.jitter_px, out.shape).astype(np.float32)
    return out


# ---------------------------------------------------------------- outputs

_SHIFT = 4  # cv2 fixed-point bits: sub-pixel accuracy of 1/16 px


def rasterize(pts: np.ndarray, end: np.ndarray, size: int = 64, pad: int = 2,
              thickness: int | None = None, order_encoding: bool = False) -> np.ndarray:
    """Draw a *normalised* drawing into a ``size x size`` uint8 image, centred
    with ``pad`` pixels of margin. Strokes of zero length are drawn as dots.

    ``order_encoding`` draws stroke i with intensity ``255 - 13 * min(i, 10)``
    so the image keeps which strokes came first (the absolute index is used, so
    a stroke has the same intensity in every prefix that contains it)."""
    thickness = thickness or max(1, round(size / 32))
    img = np.zeros((size, size), dtype=np.uint8)
    inner = size - 1 - 2 * pad
    k = inner / CANVAS
    extent = pts.max(axis=0) if len(pts) else np.zeros(2)
    offset = pad + (inner - extent * k) / 2          # centre the shorter side
    xy = np.rint((pts * k + offset) * (1 << _SHIFT)).astype(np.int32)
    bounds = np.flatnonzero(end) + 1
    for i, stroke in enumerate(np.split(xy, bounds[:-1])):
        color = 255 - 13 * min(i, 10) if order_encoding else 255
        if len(stroke) == 1:
            stroke = np.vstack([stroke, stroke])
        cv2.polylines(img, [stroke.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA, _SHIFT)
    return img


def resample_uniform(pts: np.ndarray, end: np.ndarray, step: float = RESAMPLE_STEP):
    """Evenly spaced points along each stroke of a *normalised* drawing.

    Stored coordinates sit on an integer grid, so after a prefix is rescaled the
    grid spacing reveals its true size relative to the finished drawing. Point
    spacing reveals it too. Sampling at the midpoints of ``k`` equal arcs removes
    both: unlike a scheme that keeps stroke endpoints, no original point survives
    (only dots, which have no arc). Corners can be cut by up to ``step / 2``.
    The polyline's facets remain, because they are part of the geometry."""
    out, out_end = [], []
    for s in np.split(pts.astype(np.float64), np.flatnonzero(end)[:-1] + 1):
        seg = np.hypot(*np.diff(s, axis=0).T)
        length = seg.sum()
        if length == 0:
            r = s[:1]
        else:
            k = max(2, math.ceil(length / step))
            t = np.concatenate([[0.0], np.cumsum(seg)])
            u = (np.arange(k) + 0.5) * length / k
            r = np.column_stack([np.interp(u, t, s[:, 0]), np.interp(u, t, s[:, 1])])
        e = np.zeros(len(r), dtype=bool)
        e[-1] = True
        out.append(r)
        out_end.append(e)
    return np.vstack(out).astype(np.float32), np.concatenate(out_end)


def downsample(pts: np.ndarray, end: np.ndarray, max_len: int):
    """Reduce to ``max_len`` points without cutting off the end of the drawing:
    keep every stroke's first and last point, then evenly spaced interior points."""
    n = len(pts)
    if n <= max_len:
        return pts, end
    sid = stroke_ids(end)
    starts = np.concatenate([[True], end[:-1]])
    must = np.flatnonzero(starts | end)
    if len(must) >= max_len:
        idx = np.unique(np.rint(np.linspace(0, n - 1, max_len)).astype(np.int64))
    else:
        rest = np.setdiff1d(np.arange(n), must)
        k = max_len - len(must)
        pick = rest[np.unique(np.rint(np.linspace(0, len(rest) - 1, k)).astype(np.int64))]
        idx = np.union1d(must, pick)
    kept_sid = sid[idx]
    new_end = np.concatenate([kept_sid[1:] != kept_sid[:-1], [True]])
    return pts[idx], new_end


def to_sequence(pts: np.ndarray, end: np.ndarray, max_len: int = 256,
                delta_std: tuple[float, float] = (1.0, 1.0), resample_step: float | None = None):
    """Stroke-3 sequence of a *normalised* drawing: rows ``(dx, dy, pen_lift)``
    where the first row is the offset from the origin and ``pen_lift`` = 1 on the
    last point of a stroke. dx, dy are divided by ``delta_std`` (fitted on the
    training split). ``resample_step`` applies ``resample_uniform`` first.
    Returns ``(seq[max_len, 3] float32, length)``, zero-padded."""
    if resample_step is not None:
        pts, end = resample_uniform(pts, end, resample_step)
    pts, end = downsample(pts, end, max_len)
    d = np.diff(pts.astype(np.float32), axis=0, prepend=np.zeros((1, 2), np.float32))
    seq = np.zeros((max_len, 3), dtype=np.float32)
    n = len(pts)
    seq[:n, 0] = d[:, 0] / delta_std[0]
    seq[:n, 1] = d[:, 1] / delta_std[1]
    seq[:n, 2] = end
    return seq, n
