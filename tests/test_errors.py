"""Tests for the validation error analysis (no model, no data split)."""
import numpy as np
import pytest

from quickdraw_exp.errors import answer_at, early_confusions, group_maps, stopping_by_class, within
from quickdraw_exp.hierarchy import Tree

NAMES = list("abcdef")


def _semantic():
    return Tree.from_groups({"g1": {"s1": ["a", "b"], "s2": ["c", "d"]}, "g2": ["e", "f"]}, NAMES)


def test_answer_in_force():
    ev = {"ink": np.array([0.1, 0.3, 0.6, 0.4, 0.9]), "starts": np.array([0, 3]), "ends": np.array([3, 5]),
          "probs": np.eye(6)[[0, 1, 2, 3, 4]]}
    pred, has = answer_at(ev, 0.25)
    assert has.tolist() == [True, False] and pred[0] == 0
    pred, has = answer_at(ev, 0.5)
    assert has.tolist() == [True, True] and pred.tolist() == [1, 3]


def test_errors_within_groups_and_chance():
    groups = group_maps(_semantic(), 6)[2]
    share, chance = within(np.array([0, 0, 4]), np.array([1, 4, 5]), groups)
    assert share == pytest.approx(2 / 3)
    assert chance == pytest.approx(np.mean([3 / 5, 3 / 5, 1 / 5]))


def test_confusions_and_stopping_on_synthetic_runs():
    pytest.importorskip("scipy")
    from test_analysis import _predictions

    from quickdraw_exp.analysis import _parts
    parts = [_parts(_predictions(s)) for s in (0, 1)]
    rep = early_confusions([ev for _, ev in parts], NAMES, _semantic(), _semantic(), 0.5)
    assert 0 <= rep["accuracy"] <= 1 and rep["drawings"] == 240
    assert len(rep["top_pairs"]) <= 15 and all(p["true"] != p["predicted"] for p in rep["top_pairs"] if p["share_of_true_class"] > 0)
    stop = stopping_by_class(parts, NAMES)
    assert len(stop["classes"]) == 6 and sum(r["drawings"] for r in stop["classes"]) == 240
    assert 0 < stop["overall"]["mean_ink"] <= 1
