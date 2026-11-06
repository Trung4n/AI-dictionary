"""Step 1 check: real data, logging, and resuming after a crash.

    PYTHONPATH=src python -m quickdraw_exp.smoke --config configs/smoke.yaml

Trains a tiny throwaway network three times in separate processes:
one uninterrupted run, one that saves mid-epoch and then dies, and one that
resumes it. The check passes when the uninterrupted and resumed runs end with
the same parameters. A failure exits with a non-zero code.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

CRASH_EXIT_CODE = 37


def _build(cfg: dict, device):
    import torch
    from torch import nn
    size = cfg["data"]["image_size"]
    model = nn.Sequential(nn.Flatten(), nn.Linear(size * size, 64), nn.ReLU(), nn.Dropout(0.1),
                          nn.Linear(64, 100)).to(device)
    return model, torch.optim.AdamW(model.parameters(), lr=cfg["smoke"]["lr"])


def _device(cfg: dict):
    import torch
    name = cfg["run"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _training_view(cfg: dict, root):
    """A fixed small part of the training data, without the held-out countries."""
    import numpy as np
    from quickdraw.dataset import RasterView
    from quickdraw.transforms import AugmentConfig

    from .config import repo_path
    from .data import SubsetSplit, load_holdout, open_split

    seed, smoke, data = cfg["run"]["seed"], cfg["smoke"], cfg["data"]
    holdout = load_holdout(repo_path(data["holdout_file"]))
    train = open_split("train", root, holdout)
    pick = np.random.default_rng(seed).choice(len(train), smoke["subset_size"], replace=False)
    subset = SubsetSplit(train, np.sort(pick))
    return RasterView(subset, size=data["image_size"], frac="random", augment=AugmentConfig(),
                      prefix_mode=data["prefix_mode"], frac_range=tuple(data["frac_range"]), seed=seed), holdout


def train_phase(cfg: dict, run_dir, root, resume: bool, stop_at: int | None) -> None:
    import torch

    from .checkpoint import load_checkpoint, save_checkpoint, seed_everything
    from .data import ResumableSampler
    from .runlog import RunLogger

    seed, smoke = cfg["run"]["seed"], cfg["smoke"]
    batch_size, total = smoke["batch_size"], smoke["total_steps"]
    device = _device(cfg)
    logger = RunLogger(run_dir, cfg, {"root": str(root)})
    seed_everything(seed)
    model, optimizer = _build(cfg, device)
    view, _ = _training_view(cfg, root)
    sampler = ResumableSampler(len(view), seed)
    latest = logger.checkpoint_dir / "latest.pt"
    epoch = batches_done = step = 0
    if resume:
        torch.manual_seed(987654321)  # must not matter: the checkpoint restores the generators
        torch.cuda.manual_seed_all(987654321)
        counters = load_checkpoint(latest, model, optimizer)
        epoch, batches_done, step = counters["epoch"], counters["batches_done"], counters["global_step"]
        dropped = logger.rewind(step)
        logger.event(f"resumed at step {step}, epoch {epoch}, batch {batches_done}; dropped {dropped} metric lines")

    per_epoch = len(view) // batch_size
    while step < total:
        view.set_epoch(epoch)
        sampler.set_position(epoch, batches_done * batch_size)
        # A separate generator keeps creating a loader from touching the global
        # random state, which a resumed run would otherwise see differently.
        loader = torch.utils.data.DataLoader(
            view, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=cfg["data"]["num_workers"],
            generator=torch.Generator().manual_seed(seed + epoch))
        for images, labels in loader:
            x = images.float().div(255).unsqueeze(1).to(device)
            y = torch.as_tensor(labels, dtype=torch.long).to(device)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            batches_done += 1
            logger.log(step, loss=round(float(loss.item()), 5), epoch=epoch)
            if batches_done == per_epoch:
                epoch, batches_done = epoch + 1, 0
            if stop_at is not None and step == stop_at:
                save_checkpoint(latest, model, optimizer, epoch=epoch, batches_done=batches_done, global_step=step)
                logger.event(f"saved at step {step}")
            if stop_at is not None and step == stop_at + smoke["crash_after"]:
                logger.event(f"simulating a crash at step {step}, unsaved")
                sys.stdout.flush()
                os._exit(CRASH_EXIT_CODE)
            if step >= total or batches_done == 0:
                break
    save_checkpoint(latest, model, optimizer, epoch=epoch, batches_done=batches_done, global_step=step)
    logger.event(f"finished at step {step}")


def _child(args, run_dir, *extra) -> int:
    cmd = [sys.executable, "-m", "quickdraw_exp.smoke", "--config", args.config, "--phase-dir", str(run_dir), *extra]
    for item in args.set:
        cmd += ["--set", item]
    return subprocess.run(cmd).returncode


def _parameter_difference(path_a, path_b) -> float:
    import torch
    a = torch.load(path_a, map_location="cpu", weights_only=False)["model"]
    b = torch.load(path_b, map_location="cpu", weights_only=False)["model"]
    return max(float((a[key] - b[key]).abs().max()) for key in a)


def run_check(args, cfg: dict, root, manifest: dict) -> int:
    import torch

    from .config import repo_path
    from .data import country_mask, load_holdout, open_split
    from .paths import resolve_output_root
    from .runlog import RunLogger, new_run_dir

    seed, smoke = cfg["run"]["seed"], cfg["smoke"]
    output = resolve_output_root(cfg["run"]["output_root"])
    top = new_run_dir(output, cfg["run"]["name"], seed)
    logger = RunLogger(top, cfg, {"root": str(root), "split_digest": manifest["split_digest"]})
    checks, started = {}, time.time()

    holdout = load_holdout(repo_path(cfg["data"]["holdout_file"]))
    for split in ("train", "val"):
        kept = open_split(split, root, holdout)
        checks[f"{split}_excludes_holdout"] = not bool(country_mask(kept.meta, holdout).any())
        checks[f"{split}_all_classes"] = bool(kept.meta["label"].nunique() == 100)
        logger.event(f"{split}: {len(kept):,} drawings after removing {holdout}")
    try:
        open_split("test", root, holdout)
        checks["test_locked"] = False
    except PermissionError:
        checks["test_locked"] = True

    whole, broken = top / "uninterrupted", top / "interrupted"
    checks["uninterrupted_ran"] = _child(args, whole) == 0
    checks["crash_happened"] = _child(args, broken, "--stop-at", str(smoke["stop_at"])) == CRASH_EXIT_CODE
    checks["resume_ran"] = _child(args, broken, "--resume") == 0

    difference = None
    if checks["uninterrupted_ran"] and checks["resume_ran"]:
        difference = _parameter_difference(whole / "checkpoints/latest.pt", broken / "checkpoints/latest.pt")
        checks["same_parameters_after_resume"] = difference <= smoke["tolerance"]
        steps = [m["step"] for m in RunLogger(broken, cfg).read_metrics()]
        checks["log_has_each_step_once"] = steps == list(range(1, smoke["total_steps"] + 1))
        a, b = RunLogger(whole, cfg).read_metrics(), RunLogger(broken, cfg).read_metrics()
        checks["same_losses_after_resume"] = (
            max(abs(p["loss"] - q["loss"]) for p, q in zip(a, b)) <= 1e-3)

    passed = all(checks.values())
    device = str(_device(cfg))
    summary = {"passed": passed, "checks": checks, "max_parameter_difference": difference, "device": device,
               "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               "torch": torch.__version__, "seconds": round(time.time() - started, 1)}
    logger.finish("passed" if passed else "failed", **summary)
    print(json.dumps(summary, indent=2))
    print(f"\nSTEP 1 CHECK {'PASSED' if passed else 'FAILED'}. Records: {top}")
    return 0 if passed else 1


def main(argv=None) -> int:
    from .config import load_config
    from .paths import resolve_data_root, verify_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="section.key=value")
    parser.add_argument("--phase-dir", help=argparse.SUPPRESS)
    parser.add_argument("--stop-at", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    data = cfg["data"]
    root = resolve_data_root(data["root"])
    if args.phase_dir:
        train_phase(cfg, args.phase_dir, root, args.resume, args.stop_at)
        return 0
    manifest = verify_data(root, data["split_version"], data["split_digest"], data["verify_sha256"])
    return run_check(args, cfg, root, manifest)


if __name__ == "__main__":
    sys.exit(main())
