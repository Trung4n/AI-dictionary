"""Tests for the baselines. Metric tests need NumPy only; the rest need torch."""
import numpy as np
import pandas as pd
import pytest

from quickdraw_exp.metrics import classification_metrics, expected_calibration_error, group_masks, macro_f1, softmax


# ---------------------------------------------------------------- metrics

def test_perfect_predictions():
    labels = np.array([0, 1, 2, 1])
    logits = np.eye(3)[labels] * 50.0
    m = classification_metrics(logits, labels)
    assert m["top1"] == 1.0 and m["top3"] == 1.0 and m["macro_f1"] == 1.0 and m["n"] == 4
    assert m["nll"] < 1e-6 and m["ece"] < 1e-6


def test_top3_counts_second_and_third_choice():
    logits = np.array([[3.0, 2.0, 1.0, 0.0]])
    assert classification_metrics(logits, np.array([2]))["top3"] == 1.0
    assert classification_metrics(logits, np.array([2]))["top1"] == 0.0
    assert classification_metrics(logits, np.array([3]))["top3"] == 0.0


def test_macro_f1_is_not_accuracy_on_imbalanced_data():
    labels = np.array([0] * 9 + [1])
    preds = np.zeros(10, dtype=int)
    assert (preds == labels).mean() == 0.9
    assert macro_f1(preds, labels) == pytest.approx((2 * 9 / 19 + 0) / 2)


def test_macro_f1_by_hand():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])  # class 0: p=1, r=.5, f1=2/3; class 1: p=2/3, r=1, f1=.8
    assert macro_f1(preds, labels) == pytest.approx((2 / 3 + 0.8) / 2)


def test_ece_by_hand():
    probs = np.array([[0.9, 0.1]] * 10)
    labels = np.array([0] * 7 + [1] * 3)  # accuracy .7 at confidence .9
    assert expected_calibration_error(probs, labels) == pytest.approx(0.2)


def test_ece_handles_certain_predictions():
    probs = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert expected_calibration_error(probs, np.array([0, 0])) == pytest.approx(0.0)


def test_softmax_is_stable_with_large_logits():
    p = softmax(np.array([[1000.0, 0.0]]))
    assert np.isfinite(p).all() and p[0, 0] == pytest.approx(1.0)


def test_group_masks():
    meta = pd.DataFrame({"recognized": [True, False, True], "word": ["a", "b", "c"]})
    masks = group_masks(meta, {"a": "control", "b": "confusable", "c": "control"})
    assert masks["recognized"].tolist() == [True, False, True]
    assert masks["not_recognized"].tolist() == [False, True, False]
    assert masks["role_control"].tolist() == [True, False, True]
    assert masks["role_confusable"].tolist() == [False, True, False]
    assert set(group_masks(meta)) == {"recognized", "not_recognized"}


# ---------------------------------------------------------------- models and engine

def _torch():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "randn"):
        pytest.skip("real torch needed")
    return torch


def test_cnn_shape_and_gradients():
    torch = _torch()
    from quickdraw_exp.models import RasterResNet
    model = RasterResNet(100, width=8, blocks=(1, 1))
    logits = model(torch.rand(4, 1, 64, 64))
    assert logits.shape == (4, 100)
    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_cnn_stem_stride_keeps_output_shape():
    torch = _torch()
    from quickdraw_exp.models import build_model
    model = build_model({"kind": "cnn", "width": 8, "blocks": [1, 1], "stem_stride": 2})
    assert model(torch.rand(2, 1, 64, 64)).shape == (2, 100)


def test_gru_ignores_padding():
    torch = _torch()
    from quickdraw_exp.models import CausalGRU
    model = CausalGRU(100, hidden=16, layers=2).eval()
    seq = torch.rand(1, 20, 3)
    a = model(seq, torch.tensor([5]))
    b = model(seq[:, :5], torch.tensor([5]))
    other = seq.clone()
    other[:, 5:] = 9.0
    c = model(other, torch.tensor([5]))
    assert torch.allclose(a, b) and torch.allclose(a, c)


def test_gru_is_causal():
    torch = _torch()
    from quickdraw_exp.models import CausalGRU
    model = CausalGRU(100, hidden=16, layers=2).eval()
    seq = torch.rand(2, 12, 3)
    lengths = torch.tensor([12, 12])
    before = model.forward_all_steps(seq, lengths)
    changed = seq.clone()
    changed[:, 7:] = 5.0
    after = model.forward_all_steps(changed, lengths)
    assert torch.allclose(before[:, :7], after[:, :7])
    assert not torch.allclose(before[:, 7:], after[:, 7:])


def test_last_step_output_matches_step_outputs():
    torch = _torch()
    from quickdraw_exp.models import CausalGRU
    model = CausalGRU(100, hidden=16, layers=2).eval()
    seq = torch.rand(3, 10, 3)
    lengths = torch.tensor([10, 4, 7])
    last = model(seq, lengths)
    steps = model.forward_all_steps(seq, lengths)
    for i, n in enumerate(lengths.tolist()):
        assert torch.allclose(last[i], steps[i, n - 1], atol=1e-6)


def test_to_inputs_for_both_kinds():
    torch = _torch()
    from quickdraw_exp.models import to_inputs
    images = torch.zeros(2, 64, 64, dtype=torch.uint8)
    images[0, 0, 0] = 255
    (x,), y = to_inputs("raster", (images, torch.tensor([1, 2])), "cpu")
    assert x.shape == (2, 1, 64, 64) and x.max() == 1.0 and y.dtype == torch.long
    (seq, lengths), _ = to_inputs("sequence", (torch.zeros(2, 256, 3), torch.tensor([3, 4]), torch.tensor([1, 2])), "cpu")
    assert seq.shape == (2, 256, 3) and lengths.tolist() == [3, 4]


def test_schedule_warms_up_then_decays_to_zero():
    pytest.importorskip("torch")
    from quickdraw_exp.engine import lr_lambda
    f = lr_lambda(10, 110)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(60) == pytest.approx(0.5) and f(110) == pytest.approx(0.0, abs=1e-9)


def test_stratified_subset_is_balanced_and_repeatable():
    pytest.importorskip("torch")
    from quickdraw_exp.engine import stratified_subset

    class Split:
        split, root = "val", "x"
        labels = np.repeat([0, 1, 2], 10)
        meta = pd.DataFrame({"label": labels})

    a = stratified_subset(Split, 4, seed=1)
    b = stratified_subset(Split, 4, seed=1)
    assert a.indices.tolist() == b.indices.tolist()
    assert np.bincount(a.labels).tolist() == [4, 4, 4]
    assert stratified_subset(Split, 4, seed=2).indices.tolist() != a.indices.tolist()


def test_engine_never_opens_test_split():
    source = (__import__("pathlib").Path(__file__).parents[1] / "src/quickdraw_exp/engine.py").read_text()
    assert 'open_split("test"' not in source and "final_evaluation" not in source
