"""Tests for the training-towards-convergence matrix (convergence_v1)."""
from pathlib import Path

import yaml

MATRIX = Path("configs/convergence_v1.yaml")


def _jobs(path):
    from quickdraw_exp.experiments import build_jobs
    return {j["name"]: j for j in build_jobs(yaml.safe_load(Path(path).read_text()))}


def _resolved(job):
    from quickdraw_exp.config import load_config
    cfg = load_config(job["config"], [*job["set"], f"run.seed={job['seed']}"])
    cfg["run"].pop("name", None)
    return cfg


def test_jobs():
    assert sorted(_jobs(MATRIX)) == ["gru_anytime_uniform_full_s42", "gru_anytime_uniform_full_s43",
                                     "gru_uniform_full_s42", "gru_uniform_full_s43"]


def test_only_data_and_length_differ_from_the_600k_runs():
    new = _jobs(MATRIX)
    old = {**_jobs("configs/followup_v1.yaml"), **_jobs("configs/followup_v2.yaml")}
    for seed in (42, 43):
        for mine, theirs, epochs in (("gru_uniform_full", "gru_uniform", (8, 8)),
                                     ("gru_anytime_uniform_full", "gru_anytime_uniform", (4, 4))):
            a, b = _resolved(new[f"{mine}_s{seed}"]), _resolved(old[f"{theirs}_s{seed}"])
            assert a["train"].pop("train_subset") is None and b["train"].pop("train_subset") == 600000
            assert (a["train"]["epochs"], b["train"]["epochs"]) == epochs
            assert a == b, mine


def test_comparisons_name_known_conditions():
    matrix = yaml.safe_load(MATRIX.read_text())
    known = set(matrix["conditions"]) | set(matrix["anytime"]["baseline_conditions"])
    for c in matrix["anytime"]["comparisons"]:
        assert c["a"][0] in known and c["b"][0] in known, c["name"]
