"""Tests for step 4b: hierarchical answers."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from quickdraw_exp.hierarchy import (Tree, answer, calibrate, compare, flat_tree, reward_area, reward_at, run_trees,
                                     score)

NAMES = ["a", "b", "c", "d"]


def _tree():
    return Tree.from_groups({"g1": {"s1": ["a", "b"], "s2": ["c"]}, "g2": ["d"]}, NAMES)


def test_tree_nodes_levels_and_rewards():
    t = _tree()
    assert t.names == ["g1/s1", "g1", "a", "b", "c", "d"]  # one-class nodes are dropped
    assert t.level.tolist() == [1, 2, 0, 0, 0, 0]
    assert t.reward[t.names.index("a")] == pytest.approx(1.0)
    assert t.reward[t.names.index("g1")] == pytest.approx(1 - np.log(3) / np.log(4))
    with pytest.raises(ValueError):
        Tree.from_groups({"g": ["a", "b", "c"]}, NAMES)          # d missing
    with pytest.raises(ValueError):
        Tree.from_groups({"g": ["a", "b", "c", "d", "a"]}, NAMES)  # a twice


def test_answer_is_the_most_specific_confident_node():
    t = _tree()
    probs = np.array([[0.45, 0.40, 0.10, 0.05],   # a alone 0.45; a+b 0.85
                      [0.90, 0.05, 0.03, 0.02],   # a itself is confident
                      [0.30, 0.30, 0.30, 0.10]])  # nothing below the root reaches 0.95
    nodes = answer(probs @ t.members.T, t, 0.8)
    assert [t.names[n] if n >= 0 else None for n in nodes] == ["g1/s1", "a", "g1"]
    assert answer(probs @ t.members.T, t, 0.95)[2] == -1


def test_score_rewards_correct_answers_by_specificity():
    t = _tree()
    nodes = np.array([t.names.index("g1"), t.names.index("a"), t.names.index("a"), -1])
    right, reward = score(nodes, np.array([2, 0, 1, 0]), t)
    assert right.tolist() == [True, True, False, False]
    assert reward.tolist() == pytest.approx([t.reward[t.names.index("g1")], 1.0, 0.0, 0.0])


def test_reward_area_and_reward_at_by_hand():
    reward = np.array([0.0, 0.5, 1.0, 1.0])
    ink = np.array([0.2, 0.4, 1.0, 0.5])
    starts, ends = np.array([0, 3]), np.array([3, 4])
    # drawing 0: 0 on [0.2,0.4), 0.5 on [0.4,1.0), 1 at 1.0 -> 0.3; drawing 1: 1 on [0.5,1] -> 0.5
    assert reward_area(reward, ink, starts, ends).tolist() == pytest.approx([0.3, 0.5])
    assert reward_at(reward, ink, starts, ends, 0.25).tolist() == [0.0, 0.0]
    assert reward_at(reward, ink, starts, ends, 0.5).tolist() == [0.5, 1.0]


def test_calibrate_reaches_the_target_on_its_own_data():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 2000)
    probs = rng.dirichlet(np.ones(4), 2000)
    probs[np.arange(2000), labels] += 1.0
    probs /= probs.sum(axis=1, keepdims=True)
    t = flat_tree(4)
    tau = calibrate(probs, labels, t, np.arange(0.3, 1.0, 0.01), 0.95)
    right, _ = score(answer(probs @ t.members.T, t, tau), labels, t)
    assert right.sum() / (answer(probs @ t.members.T, t, tau) >= 0).sum() >= 0.95


def test_shuffled_tree_keeps_shape_and_classes():
    t = _tree()
    s = t.shuffled(3)
    assert s.names == t.names and np.array_equal(s.members.sum(axis=1), t.members.sum(axis=1))
    assert np.array_equal(s.members[s.level == 0], t.members[t.level == 0])
    assert np.array_equal(s.members.sum(axis=0) >= 1, np.ones(4, bool))


def test_semantic_tree_beats_flat_on_partial_credit():
    rng = np.random.default_rng(1)
    n = 400
    labels = rng.integers(0, 4, n)
    probs = np.full((n, 4), 0.05)
    sibling = np.array([1, 0, 3, 2])[labels]          # unsure between a class and its sibling
    flip = np.arange(n) % 2 == 0                      # half the time the sibling looks likelier
    probs[np.arange(n), labels] = np.where(flip, 0.4, 0.5)
    probs[np.arange(n), sibling] = np.where(flip, 0.5, 0.4)
    tree = Tree.from_groups({"g1": ["a", "b"], "g2": ["c", "d"]}, NAMES)
    data = {"probs": np.vstack([probs, probs]), "label": np.concatenate([labels, labels]),
            "ink": np.full(2 * n, 0.5), "drawing": np.arange(2 * n), "half": np.repeat([0, 1], n)}
    report = run_trees(data, {"flat": [flat_tree(4)], "semantic": [tree]}, np.arange(0.3, 1.0, 0.01), [0.9])
    assert report["0.9"]["semantic"]["reward_area"] > report["0.9"]["flat"]["reward_area"]
    assert report["0.9"]["semantic"]["precision"] >= 0.9


def test_compare_needs_every_run():
    per_run = {"r1": {"0.95": {"semantic": {"reward_area": 0.4}, "flat": {"reward_area": 0.3}}}}
    rows = compare(per_run, [{"name": "x", "a": "semantic", "b": "flat", "runs": ["r1"]}], "0.95")
    assert rows[0]["difference"] == pytest.approx(0.1)


def test_frozen_tree_has_the_agreed_changes():
    groups = yaml.safe_load(Path("configs/hierarchy_v1.yaml").read_text())["groups"]
    where = {c: (g, s) for g, subs in groups.items() for s, cs in subs.items() for c in cs}
    assert where["snowman"][0] == "weather and nature" and where["umbrella"][1] == "accessory"
    assert where["megaphone"][1] == "music and sound" and where["map"][1] == "map"
    assert where["skateboard"][1] == where["rollerskates"][1] == "sport"
    assert len(where) == 100


def test_trees_build_from_the_class_list():
    import pandas as pd
    from quickdraw_exp.config import repo_path
    from quickdraw_exp.hierarchy import build_trees
    table = repo_path("reports/eda/tables/class_summary.csv")
    if not table.exists():
        pytest.skip("reports not available")
    names = sorted(pd.read_csv(table)["word"])
    trees = build_trees(yaml.safe_load(Path("configs/hierarchy_eval_v1.yaml").read_text()), names)
    assert set(trees) == {"flat", "semantic", "random", "shape"} and len(trees["random"]) == 5
    assert trees["semantic"][0].members.shape[1] == 100 and (trees["shape"][0].level <= 2).all()


def test_matrix_names_existing_runs_and_trees():
    spec = yaml.safe_load(Path("configs/hierarchy_eval_v1.yaml").read_text())
    for c in spec["comparisons"]:
        assert c["a"] in {"semantic", "random", "shape", "flat"} and set(c["runs"]) <= set(spec["runs"])
    assert spec["primary"] in [c["name"] for c in spec["comparisons"]]


def test_threshold_behaves_the_same_for_float32_and_python_floats():
    t = flat_tree(2)
    p32 = np.array([[0.5, 0.5]], dtype=np.float32)
    tau = np.float64(0.5) + 1e-15
    assert answer(p32, t, tau)[0] == answer(p32, t, float(tau))[0] == -1
