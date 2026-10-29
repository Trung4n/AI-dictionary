"""Candidate definitions of a drawing prefix ("the first k% of the drawing").

The simplified data carries no timestamps, so "how much has been drawn" must be
approximated by stroke count, point count or ink (polyline) length. All three
keep strokes in their original drawing order.
"""
from __future__ import annotations

import math

from .io import Stroke


def stroke_ink(xs: list[int], ys: list[int]) -> float:
    return sum(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(len(xs) - 1))


def prefix_by_strokes(strokes: list[Stroke], frac: float) -> list[Stroke]:
    """First floor(frac * n_strokes) whole strokes (may be empty)."""
    return strokes[: math.floor(frac * len(strokes) + 1e-9)]


def prefix_by_points(strokes: list[Stroke], frac: float) -> list[Stroke]:
    """First ceil(frac * n_points) points, cutting inside a stroke if needed."""
    budget = math.ceil(frac * sum(len(xs) for xs, _ in strokes) - 1e-9)
    out = []
    for xs, ys in strokes:
        if budget <= 0:
            break
        out.append((xs[:budget], ys[:budget]))
        budget -= len(xs)
    return out


def prefix_by_ink(strokes: list[Stroke], frac: float) -> list[Stroke]:
    """Strokes up to frac of total ink length; the last segment is interpolated."""
    target = frac * sum(stroke_ink(xs, ys) for xs, ys in strokes)
    if frac >= 1:
        return [(list(xs), list(ys)) for xs, ys in strokes]
    out, done = [], 0.0
    for xs, ys in strokes:
        nx, ny = [xs[0]], [ys[0]]
        for i in range(len(xs) - 1):
            seg = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            if done + seg >= target:
                t = (target - done) / seg if seg > 0 else 0.0
                nx.append(round(xs[i] + t * (xs[i + 1] - xs[i])))
                ny.append(round(ys[i] + t * (ys[i + 1] - ys[i])))
                out.append((nx, ny))
                return out
            done += seg
            nx.append(xs[i + 1])
            ny.append(ys[i + 1])
        out.append((nx, ny))
    return out


PREFIX_FNS = {"strokes": prefix_by_strokes, "points": prefix_by_points, "ink": prefix_by_ink}


def bbox(strokes: list[Stroke]) -> tuple[int, int, int, int] | None:
    xs = [x for s, _ in strokes for x in s]
    ys = [y for _, s in strokes for y in s]
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)
