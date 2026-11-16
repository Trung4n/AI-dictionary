"""Tests for the demo: canvas strokes in, a guess out."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from quickdraw_exp.demo import ASSETS, to_points

STROKES = [[[10, 40, 80, 120], [20, 22, 25, 60]], [[50, 52, 55], [5, 50, 90]]]


def test_points_follow_the_simplified_format_and_ignore_position_and_size():
    pts, end = to_points(STROKES)
    assert pts.dtype == np.uint8 and pts.max() == 255 and pts.min() == 0 and end.sum() == 2
    moved = [[[2 * x + 300 for x in xs], [2 * y + 7 for y in ys]] for xs, ys in STROKES]
    p2, e2 = to_points(moved)
    assert np.array_equal(pts, p2) and np.array_equal(end, e2)
    assert len(to_points([])[0]) == 0 and len(to_points([[[], []]])[0]) == 0


def test_assets_match_the_reports():
    from quickdraw_exp.hierarchy import Tree
    a = json.loads(ASSETS.read_text())
    assert len(a["classes"]) == 100 and len(set(a["classes"])) == 100
    Tree.from_groups(a["tree"], a["classes"])
    for run in ("cnn_imagenet_s42", "cnn_prefix_s42", "gru_uniform_full_s42"):
        th = a["thresholds"][run]
        assert 0 < th["stop_pmax"]["0.05"] < th["stop_pmax"]["0.02"] < 1 and 0 < th["tree"]["0.95"] < 1
    final = json.loads(Path("reports/final_v2_test_v2/results.json").read_text())["sets"]["test_v2"]
    assert a["thresholds"]["cnn_prefix_s42"]["stop_pmax"]["0.05"] == \
        final["anytime"]["per_run"]["cnn_prefix_s42"]["pmax"]["risk_control"]["0.05"]["tau"]


def test_guesser_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from quickdraw_exp.config import load_config
    from quickdraw_exp.demo import Guesser
    from quickdraw_exp.models import build_model
    run = tmp_path / "cnn_prefix_s42"
    (run / "checkpoints").mkdir(parents=True)
    cfg = load_config("configs/baseline_cnn.yaml", ["model.stem_stride=2"])
    torch.save({"model": build_model(cfg["model"]).state_dict()}, run / "checkpoints" / "best.pt")
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    g = Guesser(run)
    out = g.guess(STROKES)
    assert out["ready"] and len(out["top"]) == 5 and out["top"][0]["class"] in g.classes
    assert abs(sum(t["probability"] for t in out["top"])) <= 1.0 + 1e-6
    assert out["should_answer"] == (out["top"][0]["probability"] >= g.stop_tau)
    assert g.guess([])["ready"] is False
