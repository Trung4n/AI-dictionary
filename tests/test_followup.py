"""Tests for the follow-up runs: leak_v2 input, extra 4a seeds, longer training."""
from pathlib import Path

import numpy as np
import pytest
import yaml

FOLLOWUP = Path("configs/followup_v1.yaml")


def _jobs(path):
    from quickdraw_exp.experiments import build_jobs
    return {j["name"]: j for j in build_jobs(yaml.safe_load(Path(path).read_text()))}


def _resolved(job):
    from quickdraw_exp.config import load_config
    cfg = load_config(job["config"], [*job["set"], f"run.seed={job['seed']}"])
    cfg["run"].pop("name", None)
    return cfg


def test_followup_jobs():
    assert sorted(_jobs(FOLLOWUP)) == ["cnn_anytime_s43", "gru_anytime_long_s42", "gru_no_consistency_s43",
                                       "gru_uniform_s42", "gru_uniform_s43"]


def test_extra_seeds_use_exactly_the_anytime_v1_settings():
    new, old = _jobs(FOLLOWUP), _jobs("configs/anytime_v1.yaml")
    for condition in ("cnn_anytime", "gru_no_consistency"):
        mine = _resolved(new[f"{condition}_s43"])
        theirs = _resolved({**old[f"{condition}_s42"], "seed": 43})
        assert mine == theirs, condition


def test_uniform_differs_from_leak_v1_prefix_only_in_resampling():
    new, leak = _jobs(FOLLOWUP), _jobs("configs/leak_v1.yaml")
    for seed in (42, 43):
        mine, base = _resolved(new[f"gru_uniform_s{seed}"]), _resolved(leak[f"gru_prefix_s{seed}"])
        assert mine["data"].pop("resample_step") == 8.0 and base["data"].pop("resample_step") is None
        assert mine == base


def test_long_run_doubles_the_anytime_steps_and_nothing_else():
    new, old = _jobs(FOLLOWUP), _jobs("configs/anytime_v1.yaml")
    mine, theirs = _resolved(new["gru_anytime_long_s42"]), _resolved(old["gru_anytime_s42"])
    assert mine["train"].pop("epochs") == 2 * theirs["train"].pop("epochs")
    assert mine == theirs


def test_followup_comparisons_name_known_conditions():
    matrix = yaml.safe_load(FOLLOWUP.read_text())
    known = set(matrix["conditions"]) | set(matrix["anytime"]["baseline_conditions"])
    for c in matrix["anytime"]["comparisons"]:
        assert c["a"][0] in known and c["b"][0] in known
        assert c["a"][1] in {"pmax", "margin", "conf"} and c["b"][1] in {"pmax", "margin", "conf"}


def test_hierarchy_followup_matches_v1_except_runs():
    v1 = yaml.safe_load(Path("configs/hierarchy_eval_v1.yaml").read_text())
    new = yaml.safe_load(Path("configs/hierarchy_eval_followup_v1.yaml").read_text())
    for key in ("tree", "precision_targets", "tau_grid"):
        assert new[key] == v1[key]
    assert set(new["runs"]) <= set(_jobs(FOLLOWUP))
    for c in new["comparisons"]:
        assert set(c["runs"]) <= set(new["runs"])
    assert new["primary"] in [c["name"] for c in new["comparisons"]]


class _Base:
    split, root = "val", Path("nowhere")

    def __len__(self):
        return 1

    def __getitem__(self, i):
        xy = np.array([[0, 0], [200, 0], [200, 200], [0, 200], [0, 100], [100, 100]], dtype=np.uint8)
        return xy, np.array([False, False, False, True, False, True]), 4


def test_sequence_view_and_checkpoint_scoring_resample_the_same_way():
    pytest.importorskip("pyarrow")
    from quickdraw.dataset import load_resampled_stats
    from quickdraw.transforms import normalize, to_sequence
    from quickdraw_exp.anytime import CheckpointDataset
    from quickdraw_exp.config import load_config
    from quickdraw_exp.views import NormSequenceView

    cfg = load_config("configs/baseline_gru.yaml", ["data.resample_step=8.0"])
    scale = load_resampled_stats(8.0)
    view = NormSequenceView(_Base(), frac=None, resample_step=8.0)
    assert tuple(view.delta_std) == tuple(scale)
    seq, n, _ = view[0]
    pts, end, _ = _Base()[0]
    plain, m = to_sequence(normalize(pts), end, 256, scale)
    assert n > m  # resampling at step 8 adds points along the long edges
    items = {"drawing": np.array([0]), "k": np.array([len(pts)])}
    seq2, n2, _ = CheckpointDataset(_Base(), items, cfg, scale)[0]
    assert n2 == n and np.allclose(seq2, seq)


def test_make_view_passes_the_step_to_sequences_only():
    pytest.importorskip("pyarrow")
    pytest.importorskip("torch")
    from quickdraw_exp.config import load_config
    from quickdraw_exp.engine import make_view

    gru = load_config("configs/baseline_gru.yaml", ["data.resample_step=8.0"])
    assert make_view(gru, _Base(), 0.5, train=False).resample_step == 8.0
    plain = load_config("configs/baseline_gru.yaml")
    assert plain["data"]["resample_step"] is None
    with pytest.raises(ValueError):
        make_view(load_config("configs/baseline_cnn.yaml", ["data.resample_step=8.0"]), _Base(), 0.5, train=False)
    with pytest.raises(ValueError):
        make_view(load_config("configs/baseline_gru.yaml", ["data.resample_step=8.0", "data.input_norm=full_bbox"]),
                  _Base(), 0.5, train=False)


def test_old_run_configs_without_the_key_keep_the_stored_points(tmp_path):
    pytest.importorskip("pyarrow")
    from quickdraw_exp.anytime import sequence_scale
    root = tmp_path
    (root / "seq_stats.json").write_text('{"delta_std": [3.0, 4.0]}')
    assert tuple(sequence_scale({"data": {}}, root)) == (3.0, 4.0)
    assert tuple(sequence_scale({"data": {"resample_step": None}}, root)) == (3.0, 4.0)


# ---------------------------------------------------------------- followup_v2

FOLLOWUP2 = Path("configs/followup_v2.yaml")


def test_followup_v2_jobs():
    assert sorted(_jobs(FOLLOWUP2)) == ["gru_anytime_uniform_long_s42", "gru_anytime_uniform_s42",
                                        "gru_anytime_uniform_s43", "gru_no_consistency_uniform_s42",
                                        "gru_no_consistency_uniform_s43", "gru_uniform_long_s42"]


def test_followup_v2_repeats_anytime_v1_on_the_uniform_input():
    new, old = _jobs(FOLLOWUP2), _jobs("configs/anytime_v1.yaml")
    pairs = {"gru_anytime_uniform": "gru_anytime", "gru_no_consistency_uniform": "gru_no_consistency"}
    for mine, theirs in pairs.items():
        for seed in (42, 43):
            a = _resolved(new[f"{mine}_s{seed}"])
            b = _resolved({**old[f"{theirs}_s42"], "seed": seed})
            assert a["data"].pop("resample_step") == 8.0 and b["data"].pop("resample_step") is None
            assert a == b, (mine, seed)


def test_followup_v2_long_pair_has_equal_and_doubled_compute():
    new, v1 = _jobs(FOLLOWUP2), _jobs(FOLLOWUP)
    base = _resolved(v1["gru_uniform_s42"])
    long_base, long_method = _resolved(new["gru_uniform_long_s42"]), _resolved(new["gru_anytime_uniform_long_s42"])

    def prefixes(cfg):
        return cfg["train"]["train_subset"] * cfg["train"]["epochs"] * cfg["train"]["views_per_drawing"]
    assert prefixes(long_base) == prefixes(long_method) == 2 * prefixes(base)
    assert long_base["train"].pop("epochs") == 16 and base["train"].pop("epochs") == 8
    assert long_base == base
    method = _resolved(new["gru_anytime_uniform_s42"])
    assert long_method["train"].pop("epochs") == 2 * method["train"].pop("epochs")
    assert long_method == method


def test_followup_v2_comparisons_name_known_conditions():
    matrix = yaml.safe_load(FOLLOWUP2.read_text())
    known = set(matrix["conditions"]) | set(matrix["anytime"]["baseline_conditions"])
    assert matrix["anytime"]["baseline_conditions"] == ["gru_uniform"]
    assert "gru_uniform" in yaml.safe_load(FOLLOWUP.read_text())["conditions"]
    for c in matrix["anytime"]["comparisons"]:
        assert c["a"][0] in known and c["b"][0] in known
