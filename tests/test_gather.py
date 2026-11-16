"""Tests for the checkpoint package (no model needed for the search)."""
import json

import numpy as np
import pytest

from quickdraw_exp.gather import find_runs, find_val_predictions


def _run(root, name, finished=True, content=b"weights"):
    run = root / name
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best.pt").write_bytes(content)
    (run / "config.yaml").write_text("model: {}\n")
    (run / "summary.json").write_text(json.dumps({"finished": finished}))


def test_finished_runs_and_identical_copies(tmp_path):
    _run(tmp_path / "a", "gru_uniform_s42")
    _run(tmp_path / "b", "gru_uniform_s42")             # identical copy elsewhere: fine
    _run(tmp_path / "a", "cnn_prefix_s42", finished=False)
    found = find_runs([tmp_path])
    assert sorted(found) == ["gru_uniform_s42"]
    _run(tmp_path / "c", "gru_uniform_s42", content=b"other")
    with pytest.raises(SystemExit):
        find_runs([tmp_path])


def test_only_validation_predictions(tmp_path):
    (tmp_path / "predictions_all").mkdir()
    (tmp_path / "predictions_test").mkdir()
    np.savez(tmp_path / "predictions_all" / "gru_uniform_s42.npz", half=np.array([0, 0, 1]))
    np.savez(tmp_path / "predictions_test" / "gru_uniform_s42.npz", half=np.array([1, 1]))
    found = find_val_predictions([tmp_path])
    assert list(found) == ["gru_uniform_s42"] and found["gru_uniform_s42"].parent.name == "predictions_all"
