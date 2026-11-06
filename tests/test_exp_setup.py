"""Tests for the experiment setup. Those needing torch are skipped without it."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quickdraw_exp.config import apply_override, load_config, repo_path
from quickdraw_exp.data import (ConcatSplit, ResumableSampler, SubsetSplit, country_mask, load_holdout,
                                only_countries, open_split, open_unseen_countries, without_countries)
from quickdraw_exp.paths import REQUIRED_FILES, resolve_data_root, resolve_output_root, verify_data
from quickdraw_exp.runlog import RunLogger, adopt_run_dir, new_run_dir


# ---------------------------------------------------------------- config

def test_extends_and_override():
    cfg = load_config("configs/smoke.yaml", ["run.seed=7", "smoke.lr=0.5"])
    assert cfg["run"]["name"] == "smoke" and cfg["run"]["seed"] == 7
    assert cfg["smoke"]["lr"] == 0.5
    assert cfg["data"]["split_version"] == "v1"  # from base.yaml
    assert "extends" not in cfg


def test_override_typo_fails():
    cfg = load_config("configs/base.yaml")
    with pytest.raises(KeyError):
        apply_override(cfg, "run.sed=7")
    with pytest.raises(KeyError):
        apply_override(cfg, "rnu.seed=7")
    with pytest.raises(ValueError):
        apply_override(cfg, "run.seed")


def test_config_digest_matches_preprocessing_manifest():
    report = repo_path("reports/preprocessing/manifest.json")
    if not report.exists():
        pytest.skip("reports not available")
    assert load_config("configs/base.yaml")["data"]["split_digest"] == json.loads(report.read_text())["split_digest"]


# ---------------------------------------------------------------- paths

def _fake_root(root: Path, digest="abc", version="v1") -> Path:
    root.mkdir(parents=True)
    for name in REQUIRED_FILES:
        (root / name).write_bytes(b"x")
    files = {"seq_stats.json": __import__("hashlib").sha256(b"x").hexdigest()}
    (root / "manifest.json").write_text(json.dumps({"split_version": version, "split_digest": digest, "files": files}))
    return root


def test_explicit_root_wins_and_missing_file_is_named(tmp_path):
    root = _fake_root(tmp_path / "data")
    assert resolve_data_root(root) == root
    (root / "val_xy.npy").unlink()
    with pytest.raises(FileNotFoundError, match="val_xy.npy"):
        resolve_data_root(root)


def test_environment_variable(tmp_path, monkeypatch):
    root = _fake_root(tmp_path / "data")
    monkeypatch.setenv("QUICKDRAW_DATA", str(root))
    assert resolve_data_root() == root


def test_kaggle_search_finds_any_mount_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("QUICKDRAW_DATA", raising=False)
    monkeypatch.setattr("quickdraw_exp.paths.REPO_ROOT", tmp_path / "no_repo")
    root = _fake_root(tmp_path / "input" / "datasets" / "someone" / "quickdraw")
    assert resolve_data_root(kaggle_input=tmp_path / "input") == root
    _fake_root(tmp_path / "input" / "another")
    with pytest.raises(FileNotFoundError, match="several"):
        resolve_data_root(kaggle_input=tmp_path / "input")


def test_nothing_found_explains_itself(tmp_path, monkeypatch):
    monkeypatch.delenv("QUICKDRAW_DATA", raising=False)
    monkeypatch.setattr("quickdraw_exp.paths.REPO_ROOT", tmp_path / "no_repo")
    with pytest.raises(FileNotFoundError, match="QUICKDRAW_DATA"):
        resolve_data_root(kaggle_input=tmp_path / "input")


def test_output_root(tmp_path):
    assert resolve_output_root("x") == Path("x")
    working = tmp_path / "working"
    working.mkdir()
    assert resolve_output_root(None, working) == working / "runs"
    assert resolve_output_root(None, tmp_path / "absent").name == "runs"


def test_verify_data_rejects_other_split_and_changed_file(tmp_path):
    root = _fake_root(tmp_path / "data")
    assert verify_data(root, "v1", "abc", check_sha256=True)["split_digest"] == "abc"
    with pytest.raises(ValueError, match="different split"):
        verify_data(root, "v1", "other")
    with pytest.raises(ValueError, match="version"):
        verify_data(root, "v2", "abc")
    (root / "seq_stats.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="seq_stats"):
        verify_data(root, "v1", "abc", check_sha256=True)


# ---------------------------------------------------------------- data

class FakeSplit:
    split, root = "train", Path("nowhere")

    def __init__(self, countries, start=0):
        self.meta = pd.DataFrame({"countrycode": countries, "label": range(len(countries))})
        self.labels = self.meta["label"].to_numpy()
        self.start = start

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        return "drawing", self.start + i, int(self.labels[i])


def test_frozen_holdout_list_is_well_formed():
    assert load_holdout(repo_path("configs/country_holdout_v1.json")) == ["IN", "IT", "KR", "SA", "TH"]


def test_frozen_holdout_meets_its_rule():
    table = repo_path("reports/eda/tables/countries.csv")
    if not table.exists():
        pytest.skip("reports not available")
    t = pd.read_csv(table, keep_default_na=False)
    top10 = set(t.sort_values("n", ascending=False).head(10)["countrycode"])
    chosen = t[t["countrycode"].isin(["IN", "IT", "KR", "SA", "TH"])]
    assert (chosen["min_per_class"] >= 100).all() and not (set(chosen["countrycode"]) & top10)


def test_country_filter_keeps_missing_codes_out_of_holdout():
    base = FakeSplit(["US", "KR", None, "IT", "US"])
    assert country_mask(base.meta, ["KR", "IT"]).tolist() == [False, True, False, True, False]
    kept = without_countries(base, ["KR", "IT"])
    assert kept.meta["countrycode"].isna().sum() == 1 and len(kept) == 3
    assert kept[2][1] == 4 and kept.labels.tolist() == [0, 2, 4]
    assert len(only_countries(base, ["KR", "IT"])) == 2


def test_subset_and_concat_reach_the_right_drawings():
    a, b = FakeSplit(["KR", "US"], start=100), FakeSplit(["US", "KR", "KR"], start=200)
    all_kr = ConcatSplit([only_countries(a, ["KR"]), only_countries(b, ["KR"])], "unseen_countries")
    assert len(all_kr) == 3
    assert [all_kr[i][1] for i in range(3)] == [100, 201, 202]
    assert all_kr.split == "unseen_countries" and all_kr.labels.tolist() == [0, 1, 2]


def test_test_data_needs_explicit_request(tmp_path):
    pytest.importorskip("pyarrow")
    with pytest.raises(PermissionError):
        open_split("test", tmp_path, ["KR"])
    with pytest.raises(PermissionError):
        open_unseen_countries(tmp_path, ["KR"])


def test_sampler_is_reproducible_and_resumes_exactly():
    s = ResumableSampler(100, seed=3)
    s.set_position(2)
    whole = list(s)
    assert sorted(whole) == list(range(100))
    s.set_position(2, start=40)
    assert list(s) == whole[40:] and len(s) == 60
    s.set_position(3)
    assert list(s) != whole
    with pytest.raises(ValueError):
        s.set_position(0, start=101)


# ---------------------------------------------------------------- run records

def _config():
    return {"run": {"name": "t", "seed": 1, "account": "kiet"}}


def test_logger_records_and_rewinds(tmp_path):
    run = new_run_dir(tmp_path, "t", 1)
    log = RunLogger(run, _config(), {"split_digest": "abc"})
    for step in range(1, 6):
        log.log(step, loss=1.0 / step)
    assert log.rewind(3) == 2
    log.log(4, loss=9.0)
    assert [m["step"] for m in log.read_metrics()] == [1, 2, 3, 4]
    log.event("hello")
    log.finish("passed", value=1)
    env = json.loads((run / "env.json").read_text())
    assert env["account"] == "kiet" and env["data"]["split_digest"] == "abc" and "python" in env
    assert json.loads((run / "summary.json").read_text())["value"] == 1
    index = pd.read_csv(tmp_path / "index.csv")
    assert index.loc[0, "status"] == "passed" and index.loc[0, "account"] == "kiet"
    assert (run / "config.yaml").exists() and "hello" in (run / "events.log").read_text()


def test_resuming_keeps_original_records(tmp_path):
    run = new_run_dir(tmp_path, "t", 1)
    RunLogger(run, _config())
    first = (run / "env.json").read_text()
    RunLogger(run, {"run": {"name": "changed"}})
    assert (run / "env.json").read_text() == first and "changed" not in (run / "config.yaml").read_text()


def test_adopt_run_dir(tmp_path):
    run = new_run_dir(tmp_path / "out", "t", 1)
    RunLogger(run, _config())
    ckpt = run / "checkpoints" / "latest.pt"
    ckpt.write_bytes(b"x")
    assert adopt_run_dir(run, tmp_path / "out") == run
    assert adopt_run_dir(ckpt, tmp_path / "out") == run
    with pytest.raises(FileNotFoundError):
        adopt_run_dir(tmp_path, tmp_path / "out")


def test_time_budget():
    pytest.importorskip("torch")
    from quickdraw_exp.checkpoint import TimeBudget
    now = [0.0]
    budget = TimeBudget(1, clock=lambda: now[0])
    assert not budget.exhausted()
    now[0] = 61.0
    assert budget.exhausted()
    assert not TimeBudget(None).exhausted()


# ---------------------------------------------------------------- checkpoints (real torch)

def test_checkpoint_restores_training_exactly(tmp_path):
    torch = pytest.importorskip("torch")
    if not hasattr(torch.nn, "Linear") or not hasattr(torch, "randn"):
        pytest.skip("real torch needed")
    from quickdraw_exp.checkpoint import load_checkpoint, save_checkpoint, seed_everything

    def make():
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.5), torch.nn.Linear(8, 2))
        return model, torch.optim.AdamW(model.parameters(), lr=0.01)

    def step(model, opt):
        loss = model(torch.randn(16, 4)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    seed_everything(0)
    m1, o1 = make()
    for _ in range(5):
        step(m1, o1)
    path = save_checkpoint(tmp_path / "c" / "latest.pt", m1, o1, epoch=1, batches_done=2, global_step=5, best_metric=0.5)
    assert not list(path.parent.glob("*.tmp"))
    for _ in range(5):
        step(m1, o1)

    torch.manual_seed(999)
    m2, o2 = make()
    counters = load_checkpoint(path, m2, o2)
    assert counters == {"epoch": 1, "batches_done": 2, "global_step": 5, "best_metric": 0.5}
    for _ in range(5):
        step(m2, o2)
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(a, b)
