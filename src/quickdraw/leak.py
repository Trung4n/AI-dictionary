"""Tools for measuring how much a prefix reveals about its size in the finished drawing.

The target is ``ratio = prefix bbox long side / full drawing bbox long side``. A model
is not meant to know it at drawing time. Part of it can be inferred legitimately from
shape, so a high score alone does not prove a leak; compare arms of the same drawings.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from .transforms import stroke_ids


def side(p: np.ndarray) -> float:
    return float((p.max(axis=0) - p.min(axis=0)).max())


def approx_gcd(values, tol: float = 0.02) -> float:
    """Largest g such that every value is an integer multiple of g, within tol."""
    g = 0.0
    for v in values:
        a, b = max(abs(float(v)), g), min(abs(float(v)), g)
        while b > tol:
            a, b = b, a - b * np.floor(a / b + tol / b)
            b = max(b, 0.0)
        g = a
    return g


def grid_attack(q: np.ndarray) -> float:
    """Estimate ``ratio`` from a normalised point set: the set sits on a grid of
    spacing ``255 / s``, where s is the prefix side in stored integer coordinates and
    a finished drawing has side about 255. NaN when there is nothing to measure."""
    d = (q - q[0]).ravel()
    d = d[np.abs(d) > 0.02]
    if not len(d):
        return float("nan")
    g = approx_gcd(d)
    return 1.0 / g if g > 0.02 else float("nan")


def attack_all_points(n: np.ndarray) -> float:
    """Uses every point except the last, which a prefix interpolates off the grid."""
    return grid_attack(n[:-1] if len(n) > 2 else n)


def attack_stroke_endpoints(n: np.ndarray, end: np.ndarray) -> float:
    """Uses only stroke endpoints, which pen_lift marks in a stroke-3 sequence.
    Defeats resampling schemes that keep endpoints."""
    starts = np.concatenate([[True], end[:-1]])
    q = n[starts | end]
    return grid_attack(q[:-1] if len(q) > 2 else q)


def hit_rate(estimate, true, tol: float = 0.03) -> float:
    estimate, true = np.asarray(estimate, float), np.asarray(true, float)
    return float((np.abs(estimate / true - 1) < tol).mean())


def seq_features(n: np.ndarray, end: np.ndarray) -> list[float]:
    """Hand-made features of a normalised point sequence."""
    sid = stroke_ids(end)
    w, h = n.max(axis=0) - n.min(axis=0)
    d = np.diff(n.astype(np.float64), axis=0)
    same = sid[1:] == sid[:-1]
    seg = np.hypot(d[:, 0], d[:, 1])[same]
    v = d[same][seg > 0] if len(seg) else np.zeros((0, 2))
    if len(v) > 1:
        a = np.arctan2(v[:, 1], v[:, 0])
        turn = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
        turn_f = [*np.quantile(turn, [0.1, 0.5, 0.9]), turn.mean()]
    else:
        turn_f = [0.0] * 4
    seg_f = [*np.quantile(seg, [0, 0.1, 0.5, 0.9, 1]), seg.mean()] if len(seg) else [0.0] * 6
    return [len(n), int(end.sum()), seg.sum(), w, h, *n[0], *n[-1], *seg_f, *turn_f]


def cv_r2(X: np.ndarray, y: np.ndarray, subset: np.ndarray | None = None) -> dict:
    """5-fold gradient boosting R2 on the whole set and, optionally, on ``subset`` of
    the held-out predictions (so the model is always trained on the full range)."""
    pred = np.zeros(len(y))
    folds = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        pred[te] = HistGradientBoostingRegressor(random_state=0).fit(X[tr], y[tr]).predict(X[te])
        folds.append(r2_score(y[te], pred[te]))
    out = {"r2": float(np.mean(folds)), "r2_fold_std": float(np.std(folds))}
    if subset is not None:
        out["r2_subset"] = float(r2_score(y[subset], pred[subset]))
    return out


def cnn_r2(images: np.ndarray, y: np.ndarray, subset: np.ndarray | None = None,
           seeds=(0, 1, 2), epochs: int = 8, batch: int = 256) -> dict:
    """Small CNN regressing the target from uint8 images; fixed 85/15 hold-out,
    several initialisations."""
    import torch
    from torch import nn

    order = np.random.default_rng(0).permutation(len(y))
    n_te = len(y) // 7
    te, tr = order[:n_te], order[n_te:]
    X = torch.from_numpy(images).float().unsqueeze(1) / 255.0
    target = torch.from_numpy(y.astype(np.float32))
    scores, sub_scores = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        layers = []
        ch = 1
        for out_ch in (16, 32, 64, 128):
            layers += [nn.Conv2d(ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(), nn.MaxPool2d(2)]
            ch = out_ch
        net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * (len(tr) // batch))
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            net.train()
            perm = rng.permutation(tr)
            for k in range(len(tr) // batch):
                b = perm[k * batch:(k + 1) * batch]
                loss = nn.functional.mse_loss(net(X[b]).squeeze(1), target[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(X[te[k:k + 1024]]).squeeze(1) for k in range(0, len(te), 1024)]).numpy()
        scores.append(r2_score(y[te], pred))
        if subset is not None:
            m = subset[te]
            sub_scores.append(r2_score(y[te][m], pred[m]))
    out = {"r2": float(np.mean(scores)), "r2_seed_std": float(np.std(scores)), "seeds": len(seeds)}
    if sub_scores:
        out["r2_subset"] = float(np.mean(sub_scores))
    return out
