"""Re-implementation of the published QuickDraw "simplified" recipe, for raw strokes.

Recipe (quickdraw-dataset README): align to the top-left corner, scale uniformly to
a maximum value of 255, resample each stroke at 1 pixel spacing, then simplify with
Ramer-Douglas-Peucker at epsilon 2.0. Google's own code is not public; this version
is compared with the stored data in ``scripts/raw_pilot.py``.
"""
from __future__ import annotations

import numpy as np

EPSILON = 2.0


def _rdp_keep(p: np.ndarray, epsilon: float) -> np.ndarray:
    keep = np.zeros(len(p), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(p) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        chord = p[b] - p[a]
        norm = np.hypot(*chord)
        rel = p[a + 1:b] - p[a]
        d = np.abs(chord[0] * rel[:, 1] - chord[1] * rel[:, 0]) / norm if norm > 0 else np.hypot(*rel.T)
        i = int(np.argmax(d))
        if d[i] > epsilon:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return keep


def simplify(pts: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(pts, end)`` in any coordinate frame -> simplified integer ``(pts, end)``."""
    pts = pts.astype(np.float64)
    lo = pts.min(axis=0)
    side = float((pts.max(axis=0) - lo).max())
    pts = (pts - lo) * (255.0 / side if side > 0 else 1.0)
    out, out_end = [], []
    for s in np.split(pts, np.flatnonzero(end)[:-1] + 1):
        seg = np.hypot(*np.diff(s, axis=0).T)
        length = seg.sum()
        if length > 0:
            t = np.concatenate([[0.0], np.cumsum(seg)])
            u = np.append(np.arange(0.0, length, 1.0), length)
            s = np.column_stack([np.interp(u, t, s[:, 0]), np.interp(u, t, s[:, 1])])
            s = s[_rdp_keep(s, EPSILON)]
        else:
            s = np.vstack([s[0], s[0]])
        e = np.zeros(len(s), dtype=bool)
        e[-1] = True
        out.append(s)
        out_end.append(e)
    return np.rint(np.vstack(out)), np.concatenate(out_end)
