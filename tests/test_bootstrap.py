"""Tests for the bootstrap intervals of the final results (no model, no data split)."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from quickdraw_exp.anytime import evaluate_policies
from quickdraw_exp.bootstrap import (auc_rows, compare_rows, resample_weights, run_set, statistics, stopping_tables,
                                     tree_areas)
from quickdraw_exp.hierarchy import Tree, flat_tree, run_trees

N_CLASSES = 6
NAMES = list("abcdef")
TAUS = np.round(np.arange(0.30, 0.99 + 1e-9, 0.01), 4)


def _predictions(seed: int, strength: float = 4.0, n_drawings: int = 240, checks: int = 6) -> dict:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(N_CLASSES), n_drawings // N_CLASSES)
    rows = {k: [] for k in ("probs", "confidence", "label", "ink", "strokes", "drawing", "half")}
    for d, y in enumerate(labels):
        for j in range(checks):
            ink = (j + 1) / checks
            logits = np.random.default_rng([seed, d, j]).normal(size=N_CLASSES)
            logits[y] += strength * ink
            p = np.exp(logits) / np.exp(logits).sum()
            rows["probs"].append(p)
            rows["confidence"].append(p.max())
            rows["label"].append(y)
            rows["ink"].append(ink)
            rows["strokes"].append(j + 1)
            rows["drawing"].append(d)
            rows["half"].append(d % 2)
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["probs"] = out["probs"].astype(np.float16)
    order = np.argsort(out["half"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def _trees():
    semantic = Tree.from_groups({"g1": {"s1": ["a", "b"], "s2": ["c", "d"]}, "g2": ["e", "f"]}, NAMES)
    return {"flat": [flat_tree(N_CLASSES)], "semantic": [semantic], "random": [semantic.shuffled(s) for s in range(2)]}


def test_point_values_match_the_main_evaluation():
    data = _predictions(0)
    tables, hier = stopping_tables(data), tree_areas(data, _trees(), TAUS, [0.95, 0.9])
    n = len(tables["end_correct"])
    point = {k: float(v[0]) for k, v in statistics(tables, hier, np.ones((1, n), np.float32)).items()}
    main = evaluate_policies(data)
    for policy in ("pmax", "margin", "conf"):
        assert point[f"auc/{policy}"] == pytest.approx(main[policy]["auc"], abs=1e-12)
        for gap, rc in main[policy]["risk_control"].items():
            assert point[f"gap/{policy}/{gap}"] == pytest.approx(rc["gap"], abs=1e-12)
    assert point["end_accuracy"] == pytest.approx(main["reference"]["end_of_drawing"]["accuracy"])
    trees = run_trees(data, _trees(), TAUS, [0.95, 0.9])
    for target, by_tree in trees.items():
        for tree, r in by_tree.items():
            assert point[f"reward/{target}/{tree}"] == pytest.approx(r["reward_area"], abs=1e-12)


def test_auc_rows_matches_area():
    from quickdraw_exp.anytime import area
    rng = np.random.default_rng(1)
    e, a = rng.uniform(size=(4, 30)), rng.uniform(size=(4, 30))
    assert np.allclose(auc_rows(e, a), [area(e[i], a[i]) for i in range(4)])


def test_weights_are_counts_and_reproducible():
    first = np.concatenate(list(resample_weights(37, 120, 5)))
    again = np.concatenate(list(resample_weights(37, 120, 5)))
    assert first.shape == (120, 37) and np.array_equal(first, again)
    assert (first.sum(axis=1) == 37).all()


def test_paired_comparison_and_interval():
    per_run = {}
    for run, strength in (("good_s42", 5.0), ("good_s43", 5.0), ("weak_s42", 2.0), ("weak_s43", 2.0)):
        data = _predictions(int(strength), strength)  # same drawings and labels in every run
        per_run[run] = (stopping_tables(data), tree_areas(data, _trees(), TAUS, [0.95]))
    point, draws = run_set(per_run, 200, 0)
    for run in per_run:
        lo, hi = np.percentile(draws[run]["auc/pmax"], [2.5, 97.5])
        assert lo <= point[run]["auc/pmax"] <= hi
    row = compare_rows(point, draws, [("good vs weak", "good|auc/pmax", "weak|auc/pmax"),
                                      ("missing", "none|auc/pmax", "weak|auc/pmax")])
    assert row[0]["difference"] > 0 and row[0]["ci95"][0] > 0 and row[1]["difference"] is None


def test_runs_must_share_drawings():
    a, b = _predictions(0), _predictions(0, n_drawings=120)
    per_run = {"x_s42": (stopping_tables(a), {}), "y_s42": (stopping_tables(b), {})}
    with pytest.raises(ValueError):
        run_set(per_run, 10, 0)
