"""Tests for the input normalisations and the experiment runner."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quickdraw_exp.experiments import build_jobs, collect, decide, job_command, summarise, write_results


# ---------------------------------------------------------------- runner

def _matrix():
    return yaml.safe_load(Path("configs/leak_v1.yaml").read_text(encoding="utf-8"))


def test_matrix_expands_in_a_safe_order():
    jobs = build_jobs(_matrix())
    assert len(jobs) == 2 + 5 * 2
    assert [j["kind"] for j in jobs[:2]] == ["speed", "speed"]
    assert {j["seed"] for j in jobs[2:7]} == {42} and {j["seed"] for j in jobs[7:]} == {43}


def test_job_command_sets_seed_and_budget(tmp_path):
    job = build_jobs(_matrix())[2]
    cmd = job_command(job, tmp_path, 12.34, ["data.root=/x"])
    assert "run.seed=42" in cmd and "budget.max_minutes=12.3" in cmd and "data.root=/x" in cmd
    assert cmd[cmd.index("--run-dir") + 1] == str(tmp_path)


def test_matrix_settings_all_exist_in_their_configs():
    from quickdraw_exp.config import load_config
    matrix = _matrix()
    for job in build_jobs(matrix):
        load_config(job["config"], job["set"])  # an unknown key raises


def _scores(top1):
    return {"n": 10, "top1": top1, "top3": top1, "macro_f1": top1, "nll": 1.0, "ece": 0.1}


def _rows(condition, seed, early, late):
    fracs = {"0.1": early, "0.25": early, "0.5": early, "1.0": late}
    return [{"job": f"{condition}_s{seed}", "condition": condition, "seed": seed, "kind": "condition", "frac": f,
             **_scores(v)} for f, v in fracs.items()]


def test_summary_early_top1_ignores_late_fractions():
    table = summarise(_rows("gru_prefix", 42, 0.2, 0.9) + _rows("gru_prefix", 43, 0.4, 0.9), [0.1, 0.25, 0.5])
    assert table["gru_prefix"]["early_top1"]["mean"] == pytest.approx(0.3)
    assert table["gru_prefix"]["by_frac"]["1.0"]["top1"]["mean"] == pytest.approx(0.9)


@pytest.mark.parametrize("resampled,choice", [(0.40, "resampled"), (0.391, "resampled"), (0.389, "prefix"), (0.5, "resampled")])
def test_decision_rule_at_its_boundary(resampled, choice):
    rows = []
    for seed in (42, 43):
        rows += _rows("gru_prefix", seed, 0.40, 0.8) + _rows("gru_resampled", seed, resampled, 0.8)
    verdict = decide(summarise(rows, [0.1, 0.25, 0.5]), 0.01, 2)
    assert verdict["sequence_input"] == choice and verdict["raster_input"] == "prefix"


def test_decision_refuses_missing_seeds():
    rows = _rows("gru_prefix", 42, 0.4, 0.8) + _rows("gru_resampled", 42, 0.4, 0.8)
    assert decide(summarise(rows, [0.1]), 0.01, 2)["sequence_input"] is None


def test_decision_reports_leak_gap():
    rows = []
    for seed in (42, 43):
        rows += _rows("gru_prefix", seed, 0.3, 0.8) + _rows("gru_resampled", seed, 0.3, 0.8) + _rows("gru_full_bbox", seed, 0.5, 0.8)
    assert decide(summarise(rows, [0.1]), 0.01, 2)["gru_full_bbox_minus_prefix"] == pytest.approx(0.2)


def test_collect_skips_unfinished_and_results_are_written(tmp_path):
    import json
    jobs = [{"name": "a", "condition": "a", "seed": 1, "kind": "condition"},
            {"name": "b", "condition": "b", "seed": 1, "kind": "condition"}]
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "summary.json").write_text(json.dumps({"finished": True, "val": {"by_frac": {"1.0": _scores(0.5)}}}))
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "summary.json").write_text(json.dumps({"finished": False}))
    rows = collect(jobs, tmp_path)
    assert [r["job"] for r in rows] == ["a"]
    write_results(rows, {"x": 1}, tmp_path)
    assert pd.read_csv(tmp_path / "results.csv").loc[0, "top1"] == 0.5


def test_no_step3_code_opens_test_data():
    for name in ("experiments.py", "views.py"):
        source = (Path("src/quickdraw_exp") / name).read_text()
        assert 'open_split("test"' not in source and "open_unseen_countries" not in source


# ---------------------------------------------------------------- normalisations

@pytest.fixture
def base():
    pytest.importorskip("pyarrow")

    class Base:
        split, root = "train", Path("nowhere")

        def __len__(self):
            return 1

        def __getitem__(self, i):
            xy = np.array([[10, 10], [60, 10], [60, 40], [100, 40], [200, 240], [210, 250]], dtype=np.uint8)
            return xy, np.array([False, False, True, False, False, True]), 7

    return Base()


def _view(base, norm, **kw):
    from quickdraw_exp.views import NormSequenceView
    return NormSequenceView(base, input_norm=norm, delta_std=(1.0, 1.0), **kw)


def test_prefix_condition_is_the_existing_pipeline(base):
    from quickdraw.dataset import SequenceView
    new, _, _ = _view(base, "prefix", frac=0.3)[0]
    old, _, _ = SequenceView(base, delta_std=(1.0, 1.0), frac=0.3)[0]
    assert np.array_equal(new, old)


def test_full_bbox_keeps_the_full_frame(base):
    from quickdraw_exp.views import NormRasterView
    pts, _, _ = NormRasterView(base, input_norm="full_bbox", frac=0.3)._prepare(0)
    own, _, _ = NormRasterView(base, input_norm="prefix", frac=0.3)._prepare(0)
    side = lambda p: float((p.max(axis=0) - p.min(axis=0)).max())
    assert side(own) == pytest.approx(255, abs=1e-3)
    assert side(pts) < 200  # the short prefix stays small inside the full drawing's frame


def test_all_conditions_agree_on_the_full_drawing_without_augmentation(base):
    from quickdraw_exp.views import NormRasterView
    a, _, _ = NormRasterView(base, input_norm="full_bbox", frac=1.0)._prepare(0)
    b, _, _ = NormRasterView(base, input_norm="prefix", frac=1.0)._prepare(0)
    assert np.allclose(a, b)


def test_full_bbox_applies_the_same_random_draws(base):
    from quickdraw.transforms import AugmentConfig
    from quickdraw_exp.views import NormRasterView
    v = NormRasterView(base, input_norm="full_bbox", frac="random", augment=AugmentConfig(), seed=3)
    a, _, _ = v._prepare(0)
    b, _, _ = v._prepare(0)
    v.set_epoch(1)
    c, _, _ = v._prepare(0)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_resampling_spacing_and_stroke_structure():
    pytest.importorskip("pyarrow")
    from quickdraw_exp.views import resample_strokes
    pts = np.array([[0, 0], [100, 0], [0, 50], [0, 90]], dtype=np.float32)
    end = np.array([False, True, False, True])
    out, out_end = resample_strokes(pts, end, 10.0)
    assert out_end.sum() == 2 and out_end[-1]
    first = out[: np.flatnonzero(out_end)[0] + 1]
    assert len(first) == 11 and np.allclose(np.hypot(*np.diff(first, axis=0).T), 10.0)
    assert np.allclose(first[0], [0, 0]) and np.allclose(first[-1], [100, 0])


def test_resampling_keeps_a_dot():
    pytest.importorskip("pyarrow")
    from quickdraw_exp.views import resample_strokes
    out, out_end = resample_strokes(np.array([[5, 5], [5, 5]], dtype=np.float32), np.array([False, True]), 10.0)
    assert np.isfinite(out).all() and out_end[-1] and len(out) == len(out_end)


def test_resampled_prefix_removes_the_coordinate_lattice():
    pytest.importorskip("pyarrow")
    from quickdraw.transforms import normalize
    from quickdraw_exp.views import resample_strokes
    rng = np.random.default_rng(0)
    pts = np.cumsum(rng.integers(-9, 10, (40, 2)), axis=0)
    pts = (pts - pts.min(axis=0)).astype(np.float32)
    end = np.zeros(40, dtype=bool)
    end[-1] = True
    side = float((pts.max(axis=0) - pts.min(axis=0)).max())
    on_lattice = lambda p: np.abs(p * side / 255 - np.rint(p * side / 255)).max() < 1e-3
    assert on_lattice(normalize(pts))
    moved, moved_end = resample_strokes(normalize(pts), end, 16.0)
    assert not on_lattice(normalize(moved))


def test_unknown_norm_is_rejected(base):
    with pytest.raises(ValueError):
        _view(base, "nope", frac=0.5)
