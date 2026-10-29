"""Unit tests for the per-sample transforms (no dataset needed)."""
import math

import numpy as np
import pytest

from quickdraw.prefix import prefix_by_ink
from quickdraw.transforms import (CANVAS, AugmentConfig, augment, downsample,
                                  ink_lengths, normalize, prefix, rasterize,
                                  to_sequence)


def pack(strokes):
    pts = np.array([(x, y) for xs, ys in strokes for x, y in zip(xs, ys)], dtype=np.uint8)
    end = np.zeros(len(pts), dtype=bool)
    end[np.cumsum([len(xs) for xs, _ in strokes]) - 1] = True
    return pts, end


def random_drawing(rng, n_strokes=None):
    n_strokes = n_strokes or int(rng.integers(1, 8))
    return [(list(rng.integers(0, 256, k)), list(rng.integers(0, 256, k)))
            for k in rng.integers(2, 20, n_strokes)]


L_SHAPE = pack([([0, 10, 10], [0, 0, 10]), ([20, 20], [20, 20])])  # 20 px of ink, then a dot


# ---------------------------------------------------------------- prefix

@pytest.mark.parametrize("mode", ["ink", "points", "strokes_ceil"])
def test_full_prefix_is_identity(mode):
    pts, end = L_SHAPE
    p, e = prefix(pts, end, 1.0, mode)
    assert np.array_equal(p, pts) and np.array_equal(e, end)


@pytest.mark.parametrize("mode", ["ink", "points", "strokes_ceil"])
@pytest.mark.parametrize("frac", [0.01, 0.1, 0.25, 0.5, 0.75, 0.99])
def test_prefix_never_empty_and_ends_with_pen_lift(mode, frac):
    rng = np.random.default_rng(0)
    for _ in range(50):
        pts, end = pack(random_drawing(rng))
        p, e = prefix(pts, end, frac, mode)
        assert len(p) >= 1 and e[-1]
        assert e.sum() <= end.sum()


def test_ink_prefix_keeps_requested_ink():
    rng = np.random.default_rng(1)
    for _ in range(200):
        pts, end = pack(random_drawing(rng))
        total = ink_lengths(pts, end).sum()
        for frac in (0.1, 0.25, 0.5, 0.75):
            p, e = prefix(pts, end, frac, "ink")
            assert ink_lengths(p, e).sum() == pytest.approx(frac * total, rel=1e-9, abs=1e-9)


def test_ink_prefix_interpolates_last_segment():
    pts, end = L_SHAPE
    p, e = prefix(pts, end, 0.75, "ink")
    assert np.allclose(p, [[0, 0], [10, 0], [10, 5]]) and e.tolist() == [False, False, True]


def test_ink_prefix_matches_eda_implementation():
    """transforms.prefix (NumPy) and prefix.prefix_by_ink (lists, used in the EDA)
    must agree up to the EDA version's rounding of the interpolated point."""
    rng = np.random.default_rng(2)
    for _ in range(200):
        strokes = random_drawing(rng)
        pts, end = pack(strokes)
        for frac in (0.1, 0.25, 0.5, 0.75):
            p, _ = prefix(pts, end, frac, "ink")
            ref = [(x, y) for xs, ys in prefix_by_ink(strokes, frac) for x, y in zip(xs, ys)]
            assert len(p) == len(ref)
            assert np.abs(p - np.array(ref)).max() <= 0.5 + 1e-9


def test_dots_only_drawing_falls_back_to_points():
    pts, end = pack([([5, 5], [5, 5]), ([9, 9], [9, 9]), ([1, 1], [1, 1]), ([3, 3], [3, 3])])
    p, e = prefix(pts, end, 0.25, "ink")
    assert len(p) == math.ceil(0.25 * len(pts)) and e[-1]


def test_strokes_ceil_keeps_at_least_one_stroke():
    pts, end = pack([([0, 50, 100], [0, 50, 0])])  # single stroke
    p, _ = prefix(pts, end, 0.25, "strokes_ceil")
    assert len(p) == len(pts)
    pts, end = pack(random_drawing(np.random.default_rng(3), n_strokes=8))
    p, e = prefix(pts, end, 0.25, "strokes_ceil")
    assert e.sum() == 2


def test_prefix_rejects_bad_fraction():
    with pytest.raises(ValueError):
        prefix(*L_SHAPE, 0.0)


# ---------------------------------------------------------------- normalize / augment

def test_normalize_aligns_and_scales_long_side():
    rng = np.random.default_rng(4)
    for _ in range(100):
        pts, _ = pack(random_drawing(rng))
        n = normalize(pts * 0.3 + 17)
        assert np.allclose(n.min(axis=0), 0, atol=1e-4)
        assert (n.max(axis=0) - n.min(axis=0)).max() == pytest.approx(CANVAS, abs=1e-3)


def test_normalize_keeps_aspect_ratio():
    n = normalize(np.array([[10, 10], [110, 60]]))
    assert np.allclose(n, [[0, 0], [255, 127.5]])


def test_normalize_single_dot():
    assert np.allclose(normalize(np.array([[7, 9], [7, 9]])), 0)


def test_augment_is_reproducible_and_bounded():
    pts, _ = pack(random_drawing(np.random.default_rng(5)))
    a = augment(pts, np.random.default_rng(0))
    b = augment(pts, np.random.default_rng(0))
    assert np.array_equal(a, b)
    assert not np.allclose(a, pts)
    assert np.isfinite(a).all()


def test_augment_identity_config():
    pts, _ = pack(random_drawing(np.random.default_rng(6)))
    cfg = AugmentConfig(rotate_deg=0, scale=(1, 1), shear=0)
    assert np.allclose(augment(pts, np.random.default_rng(0), cfg), pts, atol=1e-4)


# ---------------------------------------------------------------- rasterize

def test_rasterize_shape_and_centering():
    pts, end = pack([([0, 255], [0, 0])])  # a horizontal line at the top
    img = rasterize(normalize(pts), end, size=64)
    assert img.shape == (64, 64) and img.dtype == np.uint8
    rows = np.flatnonzero(img.max(axis=1))
    assert abs(rows.mean() - 31.5) < 1.5  # shorter side is centred


def test_rasterize_draws_dots():
    pts, end = pack([([5, 5], [5, 5]), ([200, 200], [100, 100])])
    img = rasterize(normalize(pts), end, size=64)
    assert img.max() > 0
    assert len(np.unique(np.argwhere(img > 0)[:, 1] // 16)) == 2  # two separate blobs


def test_rasterize_order_encoding():
    pts, end = pack([([0, 100], [0, 0]), ([0, 100], [200, 200])])
    img = rasterize(normalize(pts), end, size=64, order_encoding=True)
    top, bottom = img[:32].max(), img[32:].max()
    assert top == 255 and bottom == 255 - 13


# ---------------------------------------------------------------- sequence

def test_downsample_keeps_stroke_ends_and_length():
    rng = np.random.default_rng(7)
    strokes = [(list(rng.integers(0, 256, 100)), list(rng.integers(0, 256, 100))) for _ in range(5)]
    pts, end = pack(strokes)
    p, e = downsample(pts, end, 64)
    assert len(p) == 64 and e.sum() == 5
    assert np.array_equal(p[-1], pts[-1]) and np.array_equal(p[0], pts[0])


def test_downsample_short_is_identity():
    pts, end = L_SHAPE
    p, e = downsample(pts, end, 256)
    assert p is pts and e is end


def test_to_sequence_roundtrip():
    pts, end = L_SHAPE
    n_pts = normalize(pts)
    seq, n = to_sequence(n_pts, end, max_len=8, delta_std=(2.0, 4.0))
    assert seq.shape == (8, 3) and n == len(pts)
    rec = np.cumsum(seq[:n, :2] * [2.0, 4.0], axis=0)
    assert np.allclose(rec, n_pts, atol=1e-3)
    assert seq[:n, 2].tolist() == end.tolist() and not seq[n:].any()
