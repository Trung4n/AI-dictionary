"""Tests for the post-experiment analyses (by recognized, hierarchy by group)."""
import numpy as np
import pytest

from quickdraw_exp.analysis import by_recognized, hierarchy_by_group, policies_on
from quickdraw_exp.anytime import evaluate_policies

N_CLASSES = 6


def _predictions(seed: int, n_drawings: int = 240, checks: int = 6, with_conf: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(N_CLASSES), n_drawings // N_CLASSES)
    rows = {k: [] for k in ("probs", "confidence", "label", "ink", "strokes", "drawing", "half")}
    for d, y in enumerate(labels):
        for j in range(checks):
            ink = (j + 1) / checks
            logits = rng.normal(size=N_CLASSES)
            logits[y] += 4.0 * ink
            p = np.exp(logits) / np.exp(logits).sum()
            rows["probs"].append(p)
            rows["confidence"].append(p.max() if with_conf else np.nan)
            rows["label"].append(y)
            rows["ink"].append(ink)
            rows["strokes"].append(j + 1)
            rows["drawing"].append(d)
            rows["half"].append(d % 2)
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["probs"] = out["probs"].astype(np.float16)
    order = np.argsort(out["half"], kind="stable")  # halves stored one after the other, drawings grouped
    return {k: v[order] for k, v in out.items()}


def test_policies_on_every_drawing_matches_the_main_evaluation():
    pytest.importorskip("scipy")
    data = _predictions(0)
    mine = policies_on(data)
    main = evaluate_policies(data)
    assert mine["pmax"]["auc"] == pytest.approx(main["pmax"]["auc"])
    assert mine["end_accuracy"] == pytest.approx(main["reference"]["end_of_drawing"]["accuracy"])


def test_subgroups_split_the_evaluation_drawings():
    pytest.importorskip("scipy")
    data = _predictions(1)
    recognized = np.arange(240) % 4 != 1  # a quarter unrecognised, all in the evaluation half
    result = by_recognized({"gru_uniform_s42": data, "gru_prefix_s42": _predictions(2)}, recognized)
    run = result["per_run"]["gru_uniform_s42"]
    assert run["recognized"]["drawings"] + run["unrecognized"]["drawings"] == run["all"]["drawings"]
    row = next(r for r in result["comparisons"] if r["name"].startswith("residual leak"))
    assert set(row) >= {"all", "recognized", "unrecognized"}
    assert row["all"]["difference"] == pytest.approx(row["all"]["a"] - row["all"]["b"])
    missing = next(r for r in result["comparisons"] if r["name"].startswith("method vs baseline, CNN"))
    assert missing["missing"]


def test_group_gains_add_up_to_the_overall_gain():
    from quickdraw_exp.hierarchy import Tree, flat_tree
    names = list("abcdef")
    semantic = Tree.from_groups({"g1": {"s1": ["a", "b"], "s2": ["c", "d"]}, "g2": ["e", "f"]}, names)
    taus = np.round(np.arange(0.30, 0.99 + 1e-9, 0.01), 4)
    result = hierarchy_by_group(_predictions(3), semantic, flat_tree(N_CLASSES), taus, names)
    groups = result["by_group"]
    assert set(groups) == {"g1", "g2"}
    total = sum(g["drawings"] for g in groups.values())
    weighted = sum(g["gain"] * g["drawings"] for g in groups.values()) / total
    assert weighted == pytest.approx(result["gain"])
    if result["gain"]:
        assert sum(g["share_of_total_gain"] for g in groups.values()) == pytest.approx(1.0)
    assert len(result["top_classes"]) == N_CLASSES and result["top_classes"][0][1] >= result["bottom_classes"][-1][1]
