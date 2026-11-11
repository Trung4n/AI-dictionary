"""Tests for the off-the-shelf base model (pretrained_v1): torchvision ResNet-18."""
from pathlib import Path

import pytest
import yaml

MATRIX = Path("configs/pretrained_v1.yaml")


def _jobs(path):
    from quickdraw_exp.experiments import build_jobs
    return {j["name"]: j for j in build_jobs(yaml.safe_load(Path(path).read_text()))}


def _resolved(job):
    from quickdraw_exp.config import load_config
    cfg = load_config(job["config"], [*job["set"], f"run.seed={job['seed']}"])
    cfg["run"].pop("name", None)
    return cfg


def test_jobs_and_seeds():
    assert sorted(_jobs(MATRIX)) == ["cnn_imagenet_s42", "cnn_imagenet_s43",
                                     "cnn_resnet18_scratch_s42", "cnn_resnet18_scratch_s43"]


def test_same_protocol_as_the_project_cnn_except_the_model():
    new, leak = _jobs(MATRIX), _jobs("configs/leak_v1.yaml")
    for seed in (42, 43):
        base = _resolved(leak[f"cnn_prefix_s{seed}"])
        for condition, pretrained in (("cnn_imagenet", True), ("cnn_resnet18_scratch", False)):
            mine = _resolved(new[f"{condition}_s{seed}"])
            model = mine.pop("model")
            assert model["kind"] == "tv_resnet" and model["arch"] == "resnet18" and model["pretrained"] is pretrained
            assert model["confidence_head"] is False
            other = {k: v for k, v in base.items() if k != "model"}
            assert mine == other, condition


def test_comparisons_name_known_conditions():
    matrix = yaml.safe_load(MATRIX.read_text())
    known = set(matrix["conditions"]) | set(matrix["anytime"]["baseline_conditions"])
    for c in matrix["anytime"]["comparisons"]:
        assert c["a"][0] in known and c["b"][0] in known, c["name"]
    hier = yaml.safe_load(Path("configs/hierarchy_eval_pretrained_v1.yaml").read_text())
    for c in hier["comparisons"]:
        assert set(c["runs"]) <= set(hier["runs"]), c["name"]
    base = yaml.safe_load(Path("configs/hierarchy_eval_v1.yaml").read_text())
    for key in ("tree", "precision_targets", "tau_grid"):
        assert hier[key] == base[key], key


def test_raster_kinds():
    pytest.importorskip("torch")
    from quickdraw_exp.models import is_raster
    assert is_raster({"kind": "cnn"}) and is_raster({"kind": "tv_resnet"}) and not is_raster({"kind": "gru"})


def test_resnet18_shapes_and_head():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from quickdraw_exp.models import build_model
    model = build_model({"kind": "tv_resnet", "arch": "resnet18", "pretrained": False, "confidence_head": True})
    assert model.input_kind == "raster" and model.backbone.feature_dim == 512
    images = torch.rand(3, 1, 64, 64)
    logits, conf = model.forward_all(images)
    assert logits.shape == (3, 100) and conf.shape == (3,)
    assert torch.allclose(model(images), logits)
    n = sum(p.numel() for p in model.backbone.parameters())
    assert 11.1e6 < n < 11.3e6
    # The ImageNet statistics are not saved with the weights.
    keys = set(model.state_dict())
    assert "backbone.mean" not in keys and "backbone.std" not in keys and "backbone.body.bn1.running_mean" in keys


def test_no_download_when_weights_are_loaded_next(monkeypatch):
    pytest.importorskip("torch")
    tv = pytest.importorskip("torchvision")
    from quickdraw_exp.models import build_model
    seen = []
    original = tv.models.resnet18
    monkeypatch.setattr(tv.models, "resnet18", lambda weights=None: seen.append(weights) or original(weights=None))
    build_model({"kind": "tv_resnet", "arch": "resnet18", "pretrained": True}, load_pretrained=False)
    assert seen == [None]
    build_model({"kind": "tv_resnet", "arch": "resnet18", "pretrained": True})
    assert seen[-1] is tv.models.ResNet18_Weights.IMAGENET1K_V1


def test_saved_weights_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from quickdraw_exp.models import build_model, load_weights
    cfg = {"kind": "tv_resnet", "arch": "resnet18", "pretrained": False}
    a = build_model(cfg).eval()
    torch.save({"model": a.state_dict()}, tmp_path / "best.pt")
    b = build_model(cfg, load_pretrained=False).eval()
    load_weights(b, torch.load(tmp_path / "best.pt", weights_only=False)["model"])
    x = torch.rand(2, 1, 64, 64)
    assert torch.allclose(a(x), b(x))
