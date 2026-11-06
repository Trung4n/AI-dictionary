"""Classification metrics in NumPy. Every model is scored by this one file."""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Mean F1 over the classes present in ``labels``. Higher is better; a class
    predicted but never present lowers its neighbours' precision, not the mean."""
    scores = []
    for c in np.unique(labels):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(scores))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Gap between confidence and accuracy, weighted over equal-width confidence
    bins. Lower is better. It can be low for a model that is simply uninformative."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    probs = softmax(logits)
    order = np.argsort(-logits, axis=1)[:, :3]
    preds = order[:, 0]
    picked = np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)
    return {
        "n": int(len(labels)),
        "top1": float((preds == labels).mean()),
        "top3": float((order == labels[:, None]).any(axis=1).mean()),
        "macro_f1": macro_f1(preds, labels),
        "nll": float(-np.log(picked).mean()),
        "ece": expected_calibration_error(probs, labels),
    }


def group_masks(meta, roles: dict | None = None) -> dict:
    """Subsets fixed before any results exist: whether Google's model recognised
    the drawing, and the class roles chosen when the classes were selected."""
    masks = {"recognized": meta["recognized"].to_numpy(dtype=bool),
             "not_recognized": ~meta["recognized"].to_numpy(dtype=bool)}
    if roles:
        role = meta["word"].map(roles).to_numpy()
        for name in sorted({r for r in roles.values()}):
            masks[f"role_{name}"] = role == name
    return masks
