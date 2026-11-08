"""Tests for the final evaluation helpers (no model, no test data)."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from quickdraw_exp.final_eval import (early_top1, find_run_dirs, leak_comparisons, reported_top1, sample_positions,
                                      with_val_calibration)


def test_sample_positions_is_stratified_capped_and_fixed():
    labels = np.array([0] * 10 + [1] * 3 + [2] * 7)
    pos = sample_positions(labels, 5, seed=2026)
    assert np.bincount(labels[pos]).tolist() == [5, 3, 5]
    assert (np.diff(pos) > 0).all() and np.array_equal(pos, sample_positions(labels, 5, seed=2026))


def test_val_calibration_comes_first_and_drawings_never_collide():
    saved = {"half": np.array([0, 0, 0, 1, 1]), "drawing": np.array([3, 3, 7, 9, 9]),
             "probs": np.eye(5, 4), "confidence": np.zeros(5), "label": np.arange(5), "ink": np.ones(5), "strokes": np.ones(5)}
    new = {"drawing": np.array([0, 0, 1]), "probs": np.ones((3, 4)), "confidence": np.ones(3), "label": np.zeros(3, int),
           "ink": np.ones(3), "strokes": np.ones(3)}
    data = with_val_calibration(saved, new)
    assert data["half"].tolist() == [0, 0, 0, 1, 1, 1]
    assert data["drawing"].tolist() == [3, 3, 7, 10, 10, 11]
    assert np.array_equal(data["probs"][3:], new["probs"])


def test_leak_comparisons_average_seeds():
    by = lambda x: {"0.1": {"top1": x}, "0.25": {"top1": x}, "0.5": {"top1": x}}
    per_run = {"gru_full_bbox_s42": by(0.3), "gru_full_bbox_s43": by(0.5), "gru_prefix_s42": by(0.2)}
    row = leak_comparisons(per_run, [{"name": "x", "a": "gru_full_bbox", "b": "gru_prefix"}], [0.1, 0.25, 0.5])[0]
    assert row["difference"] == pytest.approx(0.2) and early_top1(by(0.4), [0.1]) == 0.4
    assert leak_comparisons(per_run, [{"name": "y", "a": "cnn_prefix", "b": "gru_prefix"}], [0.1])[0]["difference"] is None


def test_find_run_dirs_needs_a_finished_run(tmp_path):
    for name, finished in (("gru_uniform_s42", True), ("gru_uniform_s43", False)):
        run = tmp_path / "a" / "runs" / name
        (run / "checkpoints").mkdir(parents=True)
        (run / "checkpoints" / "best.pt").write_bytes(b"")
        (run / "summary.json").write_text(json.dumps({"finished": finished}))
    found = find_run_dirs({"gru_uniform_s42", "gru_uniform_s43"}, [tmp_path])
    assert list(found) == ["gru_uniform_s42"]


def test_reported_top1_reads_results_csv(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("job,condition,seed,kind,frac,n,top1\ngru_prefix_s42,gru_prefix,42,condition,0.1,100,0.25\n")
    assert reported_top1([path, tmp_path / "missing.csv"]) == {"gru_prefix_s42": {"0.1": 0.25}}


def test_final_matrix_names_runs_the_experiments_trained():
    spec = yaml.safe_load(Path("configs/final_v1.yaml").read_text())
    from quickdraw_exp.experiments import build_jobs
    trained = set()
    for m in ("leak_v1", "anytime_v1", "followup_v1", "followup_v2"):
        trained |= {j["name"] for j in build_jobs(yaml.safe_load(Path(f"configs/{m}.yaml").read_text())) if j["kind"] == "condition"}
    for key in ("leak_runs", "anytime_runs", "hierarchy_runs", "unseen_runs"):
        assert set(spec[key]) <= trained, key
    assert set(spec["unseen_runs"]) <= set(spec["anytime_runs"]) | set(spec["leak_runs"])
    h = yaml.safe_load(Path(spec["hierarchy_matrix"]).read_text())
    for c in spec["hierarchy_comparisons"]:
        assert set(c["runs"]) <= set(spec["hierarchy_runs"]) and c["a"] in {"semantic", "random", "shape", "flat"}
    assert h["precision_targets"] == [0.95, 0.90]


def test_only_the_final_evaluation_opens_test_data():
    for path in Path("src/quickdraw_exp").glob("*.py"):
        if path.name != "final_eval.py":
            assert "final_evaluation=True" not in path.read_text(), path.name
