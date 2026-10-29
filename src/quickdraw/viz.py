"""Plotting and a minimal NumPy rasterizer for drawings in stroke format."""
from __future__ import annotations

import math

import numpy as np
from matplotlib.collections import LineCollection

from .io import Stroke
from .prefix import bbox


def plot_strokes(ax, strokes: list[Stroke], order_colors: bool = False, lw: float = 1.2,
                 title: str | None = None, frame: bool = True, canvas: int = 255):
    """Draw strokes in image coordinates (y down). ``order_colors`` colours strokes
    from dark (first) to light (last) so the drawing order is visible."""
    segs, colors = [], []
    n = max(len(strokes), 1)
    for i, (xs, ys) in enumerate(strokes):
        pts = np.column_stack([xs, ys])
        if len(pts) == 1:
            ax.plot(pts[0, 0], pts[0, 1], "k.", ms=2)
            continue
        segs.append(pts)
        colors.append(_order_color(i, n) if order_colors else "black")
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw))
    ax.set_xlim(-5, canvas + 5)
    ax.set_ylim(canvas + 5, -5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(frame)
        s.set_color("0.8")
    if title:
        ax.set_title(title, fontsize=7)


def _order_color(i: int, n: int):
    import matplotlib

    return matplotlib.colormaps["viridis"](0.85 * i / max(n - 1, 1))


def renormalize(strokes: list[Stroke], canvas: int = 255) -> list[Stroke]:
    """Shift to the top-left corner and scale the longer side to ``canvas``,
    keeping the aspect ratio (the Kaggle "simplified" convention)."""
    b = bbox(strokes)
    if b is None:
        return []
    x0, y0, x1, y1 = b
    scale = canvas / max(x1 - x0, y1 - y0, 1)
    return [([(x - x0) * scale for x in xs], [(y - y0) * scale for y in ys]) for xs, ys in strokes]


def rasterize(strokes: list[Stroke], size: int = 32, pad: int = 2,
              ref: list[Stroke] | None = None) -> np.ndarray:
    """Binary ``size x size`` image of the strokes, sampling each segment densely.

    The bbox of ``ref`` (default: the strokes themselves) is scaled to fit
    ``size - 2 * pad`` pixels and centred. Passing the full drawing as ``ref``
    renders a prefix in the full drawing's frame instead of its own."""
    img = np.zeros((size, size), dtype=np.float32)
    b = bbox(ref if ref is not None else strokes)
    if b is None:
        return img
    x0, y0, x1, y1 = b
    inner = size - 1 - 2 * pad
    k = inner / max(x1 - x0, y1 - y0, 1)
    ox = pad + (inner - (x1 - x0) * k) / 2
    oy = pad + (inner - (y1 - y0) * k) / 2
    for xs, ys in strokes:
        x = (np.asarray(xs, dtype=np.float64) - x0) * k + ox
        y = (np.asarray(ys, dtype=np.float64) - y0) * k + oy
        if len(x) == 1:
            px, py = x, y
        else:
            steps = np.maximum(np.ceil(np.hypot(np.diff(x), np.diff(y)) * 2), 1).astype(int)
            t = np.concatenate([np.linspace(0, 1, s, endpoint=False) for s in steps] + [[1.0]])
            seg = np.concatenate([np.full(s, i) for i, s in enumerate(steps)] + [[len(x) - 2]])
            px = x[seg] + t * (x[seg + 1] - x[seg])
            py = y[seg] + t * (y[seg + 1] - y[seg])
        img[np.clip(np.rint(py), 0, size - 1).astype(int), np.clip(np.rint(px), 0, size - 1).astype(int)] = 1
    return img


def mean_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / d if d else float("nan")
