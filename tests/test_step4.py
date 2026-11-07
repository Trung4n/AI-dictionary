"""Tests for step 4a: anytime evaluation, multi-prefix training, confidence head."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from quickdraw_exp.anytime import (area, build_items, calibrate, calibration_split, checkpoints, compare, frontier,
                                   group_bounds, halt_rows, ink_fraction, sweep)


# ---------------------------------------------------------------- checkpoints

def test_checkpoints_are_causal_and_end_with_the_drawing():
    end = np.zeros(20, dtype=bool)
    end[[2, 11, 19]] = True  # strokes end after points 3, 12, 20
    ks = checkpoints(end, every=8)
    assert ks.tolist() == [3, 8, 12, 16, 20]
    assert checkpoints(np.array([True])).tolist() == [1]


def test_checkpoints_are_thinned_but_keep_the_end():
    end = np.zeros(1000, dtype=bool)
    end[-1] = True
    ks = checkpoints(end, every=8, max_checks=40)
    assert len(ks) == 40 and ks[-1] == 1000 and (np.diff(ks) > 0).all()


def test_ink_fraction_ignores_pen_lifts():
    pts = np.array([[0, 0], [10, 0], [100, 100], [100, 130]])
    end = np.array([False, True, False, True])  # 10 units, a jump, then 30 units
    assert ink_fraction(pts, end, np.array([2, 3, 4])).tolist() == pytest.approx([0.25, 0.25, 1.0])
    dots = np.array([[5, 5], [5, 5]])
    assert ink_fraction(dots, np.array([False, True]), np.array([1, 2])).tolist() == [0.5, 1.0]


def test_build_items_groups_rows_by_drawing():
    class Split:
        def __getitem__(self, i):
            end = np.zeros(10, dtype=bool)
            end[[4, 9]] = True
            return np.arange(20).reshape(10, 2), end, i

    items = build_items(Split(), [3, 7])
    starts, ends = group_bounds(items["drawing"])
    assert starts.tolist() == [0, 3] and ends.tolist() == [3, 6]
    assert items["k"].tolist() == [5, 8, 10, 5, 8, 10] and items["strokes"].tolist() == [1, 1, 2, 1, 1, 2]
    assert items["label"].tolist() == [3, 3, 3, 7, 7, 7]


def test_calibration_split_is_balanced_disjoint_and_fixed():
    labels = np.repeat(np.arange(5), 300)
    a, b = calibration_split(labels, per_class=100)
    assert len(set(a) & set(b)) == 0 and np.bincount(labels[a]).tolist() == [100] * 5
    assert calibration_split(labels, per_class=100)[0].tolist() == a.tolist()


# ---------------------------------------------------------------- policies

def _toy():
    # drawing 0: right from the second check; drawing 1: right only at the end
    score = np.array([0.2, 0.9, 0.95, 0.3, 0.4, 0.6])
    correct = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    ink = np.array([0.2, 0.5, 1.0, 0.3, 0.6, 1.0])
    starts, ends = np.array([0, 3]), np.array([3, 6])
    return score, correct, ink, starts, ends


def test_halt_rows_first_true_or_last():
    starts, ends = np.array([0, 3]), np.array([3, 5])
    mask = np.array([False, True, True, False, False])
    assert halt_rows(mask, starts, ends).tolist() == [1, 4]


def test_sweep_by_hand():
    score, correct, ink, starts, ends = _toy()
    out = sweep(score, correct, ink, starts, ends, [-np.inf, 0.5, np.inf])
    assert out["accuracy"].tolist() == [0.0, 1.0, 1.0]
    assert out["earliness"].tolist() == pytest.approx([0.25, 0.75, 1.0])
    assert out["pictionary"].tolist() == pytest.approx([0.0, 0.25, 0.0])


def test_frontier_and_area():
    e, a = np.array([0.25, 0.75, 1.0]), np.array([0.0, 1.0, 1.0])
    f = frontier(e, a, np.array([0.2, 0.5, 0.75, 1.0]))
    assert f.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert area(e, a) == pytest.approx(26 / 101)  # grid points 0.75 ... 1.00 reach accuracy 1


def test_area_rewards_earlier_answers():
    late = area(np.array([0.9]), np.array([0.8]))
    early = area(np.array([0.5]), np.array([0.8]))
    assert early > late


def test_calibrated_gap_holds_on_fresh_data():
    pytest.importorskip("scipy")
    from quickdraw_exp.anytime import accuracy_gap
    rng = np.random.default_rng(0)

    def make(n):
        score = np.sort(rng.random((n, 5)), axis=1)               # confidence grows as the drawing goes on
        correct = (rng.random((n, 5)) < 0.2 + 0.75 * score).astype(float)
        correct[:, -1] = np.maximum(correct[:, -1], rng.random(n) < 0.8)
        ink = np.tile(np.linspace(0.2, 1, 5), n)
        starts = np.arange(n) * 5
        return score.ravel(), correct.ravel(), ink, starts, starts + 5

    cal, ev = make(4000), make(4000)
    taus = np.linspace(0, 1, 51)
    tau = calibrate(*cal, taus, gap=0.05)
    assert 0 < tau < 1
    rows = halt_rows(ev[0] >= tau, ev[3], ev[4])
    assert accuracy_gap(ev[1], rows, ev[4]).mean() <= 0.05 + 0.01
    assert ev[2][rows].mean() < 1.0                              # it does answer early
    assert calibrate(*cal, taus, gap=0.0) == np.inf               # no early answer is allowed


def test_gap_counts_only_answers_the_end_would_get_right():
    from quickdraw_exp.anytime import accuracy_gap
    correct = np.array([0.0, 1.0, 1.0, 0.0])
    ends = np.array([2, 4])
    assert accuracy_gap(correct, np.array([0, 2]), ends).tolist() == [1.0, 0.0]


def test_compare_reports_missing_conditions():
    table = {"x": {"conf": {"auc": {"mean": 0.6, "values": [0.6]}}},
             "y": {"pmax": {"auc": {"mean": 0.5, "values": [0.5]}}}}
    rows = compare(table, [{"name": "a", "a": ["x", "conf"], "b": ["y", "pmax"]},
                           {"name": "b", "a": ["z", "conf"], "b": ["y", "pmax"]}])
    assert rows[0]["difference"] == pytest.approx(0.1) and rows[1]["difference"] is None


def test_matrix_settings_exist_and_budget_matches_baseline():
    from quickdraw_exp.config import load_config
    from quickdraw_exp.experiments import build_jobs
    matrix = yaml.safe_load(Path("configs/anytime_v1.yaml").read_text())
    jobs = build_jobs(matrix)
    assert sorted(j["name"] for j in jobs) == ["cnn_anytime_s42", "gru_anytime_s42", "gru_anytime_s43",
                                               "gru_no_consistency_s42"]
    leak = load_config("configs/baseline_gru.yaml", yaml.safe_load(Path("configs/leak_v1.yaml").read_text())["common"])
    for job in jobs:
        cfg = load_config(job["config"], job["set"])
        prefixes = cfg["train"]["train_subset"] * cfg["train"]["epochs"] * cfg["train"]["views_per_drawing"]
        assert prefixes == leak["train"]["train_subset"] * leak["train"]["epochs"]
        assert cfg["train"]["batch_size"] == leak["train"]["batch_size"]
        assert cfg["eval"]["subset_per_class"] == leak["eval"]["subset_per_class"]
    named = {c for comp in matrix["anytime"]["comparisons"] for c in (comp["a"][0], comp["b"][0])}
    assert named <= set(matrix["conditions"]) | set(matrix["anytime"]["baseline_conditions"])


def test_no_step4_code_opens_test_data():
    source = Path("src/quickdraw_exp/anytime.py").read_text()
    assert 'open_split("test"' not in source and "open_unseen_countries" not in source


# ---------------------------------------------------------------- multi-prefix view

def test_multi_prefix_view_orders_prefixes_and_shares_augmentation():
    pytest.importorskip("pyarrow")
    from quickdraw.transforms import AugmentConfig
    from quickdraw_exp.views import MultiPrefixView, NormSequenceView

    class Base:
        split, root = "train", Path("nowhere")

        def __len__(self):
            return 1

        def __getitem__(self, i):
            xy = np.array([[0, 0], [50, 0], [50, 50], [0, 50], [0, 100], [100, 100]], dtype=np.uint8)
            return xy, np.array([False, False, False, True, False, True]), 4

    view = NormSequenceView(Base(), frac="random", augment=AugmentConfig(), delta_std=(1.0, 1.0), seed=1)
    multi = MultiPrefixView(view, 2)
    seqs, lengths, label = multi[0]
    assert seqs.shape == (2, 256, 3) and label == 4 and lengths[0] <= lengths[1]
    again = multi[0]
    assert np.array_equal(again[0], seqs) and view.frac == "random"
    with pytest.raises(ValueError):
        MultiPrefixView(NormSequenceView(Base(), frac=0.5, delta_std=(1.0, 1.0)), 2)


# ---------------------------------------------------------------- models (real torch)

def _torch():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "randn"):
        pytest.skip("real torch needed")
    return torch


def test_confidence_head_does_not_train_the_classifier():
    torch = _torch()
    from quickdraw_exp.models import anytime_loss, build_model
    model = build_model({"kind": "gru", "hidden": 16, "layers": 1, "dropout": 0.0, "confidence_head": True})
    seq, lengths, labels = torch.rand(4, 10, 3), torch.tensor([10, 6, 8, 3]), torch.tensor([0, 1, 2, 3])
    logits, conf = model.forward_all(seq, lengths)
    assert conf.shape == (4,) and ((conf >= 0) & (conf <= 1)).all()
    _, parts = anytime_loss(logits, labels, conf, 1, 0.0, 1.0)
    torch.autograd.backward(parts["conf"].requires_grad_() if False else
                            torch.nn.functional.mse_loss(conf, torch.softmax(logits.detach(), 1)[:, 0]))
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is not None for p in model.confidence.parameters())


def test_consistency_term_is_zero_when_prefixes_agree():
    torch = _torch()
    from quickdraw_exp.models import anytime_loss
    logits = torch.randn(3, 1, 100).repeat(1, 2, 1).reshape(6, 100)
    labels = torch.tensor([1, 1, 2, 2, 3, 3])
    _, parts = anytime_loss(logits, labels, None, 2, 0.5, 1.0)
    assert float(parts["kl"]) == pytest.approx(0.0, abs=1e-6)
    different = torch.randn(6, 100)
    assert float(anytime_loss(different, labels, None, 2, 0.5, 1.0)[1]["kl"]) > 0


def test_consistency_teacher_is_the_later_prefix():
    torch = _torch()
    from quickdraw_exp.models import anytime_loss
    logits = torch.randn(2, 100, requires_grad=True)
    loss, _ = anytime_loss(logits, torch.tensor([5, 5]), None, 2, 1.0, 0.0)
    loss_kl = loss - torch.nn.functional.cross_entropy(logits, torch.tensor([5, 5]))
    loss_kl.backward()
    assert logits.grad[1].abs().sum() == 0 and logits.grad[0].abs().sum() > 0


def test_to_inputs_lays_out_prefixes_by_drawing():
    torch = _torch()
    from quickdraw_exp.models import to_inputs
    seq = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32).reshape(2, 2, 4, 3)
    (flat, lengths), labels = to_inputs("sequence", (seq, torch.tensor([[1, 2], [3, 4]]), torch.tensor([7, 9])), "cpu", views=2)
    assert flat.shape == (4, 4, 3) and torch.equal(flat[1], seq[0, 1])
    assert lengths.tolist() == [1, 2, 3, 4] and labels.tolist() == [7, 7, 9, 9]


def test_leak_v1_weights_load_into_the_wrapper():
    torch = _torch()
    from quickdraw_exp.models import CausalGRU, build_model, load_weights
    old = CausalGRU(100, hidden=16, layers=1)
    model = build_model({"kind": "gru", "hidden": 16, "layers": 1, "dropout": 0.0})
    load_weights(model, old.state_dict())
    x, n = torch.rand(2, 5, 3), torch.tensor([5, 3])
    assert torch.allclose(model(x, n), old(x, n))


def test_hierarchy_draft_covers_every_class_once():
    import pandas as pd
    from quickdraw_exp.config import repo_path
    table = repo_path("reports/eda/tables/class_summary.csv")
    if not table.exists():
        pytest.skip("reports not available")
    tree = yaml.safe_load(Path("configs/hierarchy_v1.yaml").read_text())["groups"]
    leaves = [c for group in tree.values() for sub in group.values() for c in sub]
    assert len(leaves) == len(set(leaves)) == 100
    assert set(leaves) == set(pd.read_csv(table)["word"])


def test_policy_scores_are_float64():
    from quickdraw_exp.anytime import scores_from
    s = scores_from(np.array([[0.6, 0.4]], dtype=np.float16), np.array([0.5], dtype=np.float32))
    assert all(v.dtype == np.float64 for v in s.values())
