"""Training and evaluation shared by every model.

    PYTHONPATH=src python -m quickdraw_exp.engine --config configs/baseline_cnn.yaml
    PYTHONPATH=src python -m quickdraw_exp.engine --config configs/baseline_cnn.yaml --resume <run folder>

Models are chosen on validation data. This file never opens the test split.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import pandas as pd
import torch

from .checkpoint import TimeBudget, load_checkpoint, save_checkpoint, seed_everything
from .config import load_config, repo_path
from .data import ResumableSampler, SubsetSplit, load_holdout, open_split
from .metrics import classification_metrics, group_masks
from .models import anytime_loss, build_model, is_raster, to_inputs
from .paths import resolve_data_root, resolve_output_root, verify_data
from .runlog import RunLogger, adopt_run_dir, new_run_dir


def stratified_subset(split, per_class: int, seed: int) -> SubsetSplit:
    """Up to ``per_class`` drawings of each class, the same ones on every run."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(split.labels)
    picked = []
    for c in np.unique(labels):
        where = np.flatnonzero(labels == c)
        picked.append(rng.permutation(where)[:per_class])
    return SubsetSplit(split, np.sort(np.concatenate(picked)))


def make_view(cfg: dict, base, frac, train: bool):
    """Preprocessing for one model: the only place the two input kinds differ."""
    from quickdraw.transforms import AugmentConfig

    from .views import NormRasterView, NormSequenceView

    data = cfg["data"]
    common = dict(frac=frac, prefix_mode=data["prefix_mode"], augment=AugmentConfig() if train else None,
                  frac_range=tuple(data["frac_range"]), seed=cfg["run"]["seed"],
                  input_norm=data["input_norm"], spacing=data["resample_spacing"])
    step = data.get("resample_step")
    if is_raster(cfg["model"]):
        if step is not None:
            raise ValueError("data.resample_step applies to sequences only")
        return NormRasterView(base, size=data["image_size"], order_encoding=data["order_encoding"], **common)
    if step is not None and data["input_norm"] != "prefix":
        raise ValueError("data.resample_step needs input_norm=prefix")
    # With a step, SequenceView loads the delta scale fitted for resampled sequences.
    return NormSequenceView(base, max_len=data["max_len"], resample_step=step, **common)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device.type, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def predict(model, cfg: dict, base, frac, device, amp: bool) -> np.ndarray:
    """Logits for every drawing of ``base``, in order, at one fraction."""
    view = make_view(cfg, base, frac, train=False)
    loader = torch.utils.data.DataLoader(view, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                                         num_workers=cfg["data"]["num_workers"], pin_memory=device.type == "cuda")
    was_training = model.training
    model.eval()
    parts = []
    for batch in loader:
        inputs, _ = to_inputs(model.input_kind, batch, device)
        with _autocast(device, amp):
            parts.append(model(*inputs).float().cpu().numpy())
    model.train(was_training)
    return np.concatenate(parts)


def evaluate(model, cfg: dict, base, device, roles: dict | None = None, with_groups: bool = False) -> dict:
    """The one evaluation used for every model: same fractions, same metrics.
    ``mean_top1`` over the fractions is what selects the best checkpoint."""
    amp = cfg["train"]["amp"]
    labels = np.asarray(base.labels)
    masks = group_masks(base.meta, roles) if with_groups else {}
    by_frac = {}
    for frac in cfg["eval"]["fracs"]:
        logits = predict(model, cfg, base, frac, device, amp)
        scores = classification_metrics(logits, labels)
        if masks:
            scores["groups"] = {name: classification_metrics(logits[mask], labels[mask])
                                for name, mask in masks.items() if mask.any()}
        by_frac[str(frac)] = scores
    return {"by_frac": by_frac, "mean_top1": float(np.mean([s["top1"] for s in by_frac.values()]))}


def load_roles() -> dict | None:
    """Class roles from the EDA, if the reports are in this checkout."""
    path = repo_path("reports/eda/tables/class_summary.csv")
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    return dict(zip(table["word"], table["role"]))


def lr_lambda(warmup: int, total: int):
    """Linear warm-up, then cosine decay to zero, stepped per batch."""
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return factor


def open_data(cfg: dict):
    data = cfg["data"]
    root = resolve_data_root(data["root"])
    manifest = verify_data(root, data["split_version"], data["split_digest"], data["verify_sha256"])
    holdout = load_holdout(repo_path(data["holdout_file"]))
    train = open_split("train", root, holdout)
    val = open_split("val", root, holdout)
    seed = cfg["run"]["seed"]
    if cfg["train"]["train_subset"] is not None:
        per_class = max(1, cfg["train"]["train_subset"] // int(train.meta["label"].nunique()))
        train = stratified_subset(train, per_class, seed)
    info = {"root": str(root), "split_digest": manifest["split_digest"], "holdout": holdout,
            "n_train": len(train), "n_val": len(val)}
    return train, val, info


def run_training(cfg: dict, run_dir=None) -> dict:
    """Train until the epochs or the time budget end. ``run_dir`` holding a
    checkpoint means continue it."""
    seed, tcfg = cfg["run"]["seed"], cfg["train"]
    seed_everything(seed)
    device = pick_device(cfg["run"]["device"])
    train_split, val_split, info = open_data(cfg)
    if run_dir is None:
        run_dir = new_run_dir(resolve_output_root(cfg["run"]["output_root"]), cfg["run"]["name"], seed)
    logger = RunLogger(run_dir, cfg, info)
    latest, best = logger.checkpoint_dir / "latest.pt", logger.checkpoint_dir / "best.pt"

    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    # batch_size counts prefixes, so more prefixes per drawing means fewer drawings
    # per step at the same cost per step.
    views = tcfg["views_per_drawing"]
    if tcfg["batch_size"] % views:
        raise ValueError("train.batch_size must be a multiple of train.views_per_drawing")
    batch_size = tcfg["batch_size"] // views
    steps_per_epoch = len(train_split) // batch_size
    if steps_per_epoch == 0:
        raise ValueError(f"{len(train_split)} training drawings is less than one batch of {batch_size}")
    total_steps = steps_per_epoch * tcfg["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(tcfg["warmup_steps"], total_steps))
    amp = tcfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    epoch = batches_done = step = 0
    best_metric = None
    if latest.is_file():
        counters = load_checkpoint(latest, model, optimizer, scheduler, scaler)
        epoch, batches_done, step = counters["epoch"], counters["batches_done"], counters["global_step"]
        best_metric = counters["best_metric"]
        dropped = logger.rewind(step)
        logger.event(f"resumed at step {step}, epoch {epoch}, batch {batches_done}; dropped {dropped} metric lines")

    n_params = sum(p.numel() for p in model.parameters())
    logger.event(f"{cfg['model']['kind']}: {n_params:,} parameters on {device}, "
                 f"{len(train_split):,} train, {len(val_split):,} val, {total_steps} steps")
    view = make_view(cfg, train_split, "random", train=True)
    if views > 1:
        from .views import MultiPrefixView
        view = MultiPrefixView(view, views)
    sampler = ResumableSampler(len(view), seed)
    val_small = stratified_subset(val_split, cfg["eval"]["subset_per_class"], seed)

    def checkpoint(path):
        save_checkpoint(path, model, optimizer, epoch=epoch, batches_done=batches_done, global_step=step,
                        scheduler=scheduler, scaler=scaler, best_metric=best_metric)

    budget = TimeBudget(cfg["budget"]["max_minutes"])
    save_every = 60.0 * cfg["checkpoint"]["every_minutes"]
    last_save, stopped = time.monotonic(), False
    window_start, window_steps, rates, first_losses, last_losses = time.monotonic(), 0, [], [], []
    model.train()
    while epoch < tcfg["epochs"] and not stopped:
        view.set_epoch(epoch)
        sampler.set_position(epoch, batches_done * batch_size)
        # A separate generator keeps creating a loader from touching the global random state.
        loader = torch.utils.data.DataLoader(
            view, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=cfg["data"]["num_workers"],
            pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed + epoch))
        for batch in loader:
            inputs, labels = to_inputs(model.input_kind, batch, device, non_blocking=True, views=views)
            with _autocast(device, amp):
                logits, confidence = model.forward_all(*inputs)
            loss, parts = anytime_loss(logits, labels, confidence, views,
                                       tcfg["consistency_weight"], tcfg["confidence_weight"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            batches_done += 1
            window_steps += 1
            value = float(loss.item())
            if len(first_losses) < 20:
                first_losses.append(value)
            last_losses = (last_losses + [value])[-20:]
            if step % tcfg["log_every"] == 0:
                now = time.monotonic()
                rate = window_steps * batch_size / max(now - window_start, 1e-9)
                rates.append(rate)
                logger.log(step, loss=round(value, 5), lr=scheduler.get_last_lr()[0], epoch=epoch, samples_per_s=round(rate, 1),
                           **{k: round(float(v), 5) for k, v in parts.items()})
                window_start, window_steps = now, 0
            if budget.exhausted():
                stopped = True
                break
            if time.monotonic() - last_save >= save_every:
                checkpoint(latest)
                last_save = time.monotonic()
        if stopped:
            break
        epoch, batches_done = epoch + 1, 0
        scores = evaluate(model, cfg, val_small, device)
        logger.log(step, val_mean_top1=round(scores["mean_top1"], 5), epoch=epoch)
        logger.event(f"epoch {epoch}: val mean top-1 {scores['mean_top1']:.4f}")
        window_start, window_steps = time.monotonic(), 0
        if best_metric is None or scores["mean_top1"] > best_metric:
            best_metric = scores["mean_top1"]
            checkpoint(best)
        checkpoint(latest)
        last_save = time.monotonic()

    checkpoint(latest)
    summary = {"finished": not stopped, "steps": step, "views_per_drawing": views, "epochs_done": epoch, "parameters": n_params,
               "device": str(device), "best_val_mean_top1": best_metric,
               "first_loss": float(np.mean(first_losses)) if first_losses else None,
               "last_loss": float(np.mean(last_losses)) if last_losses else None,
               "train_samples_per_s": float(np.median(rates)) if rates else None,
               "n_train": info["n_train"], "n_val": info["n_val"]}
    if device.type == "cuda":
        summary["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 20)
    if not stopped and best.is_file():
        load_checkpoint(best, model)
        final = cfg["eval"]["final_per_class"]
        target = val_split if final is None else stratified_subset(val_split, final, seed)
        summary["val"] = evaluate(model, cfg, target, device, load_roles(), with_groups=True)
    logger.event("stopped by the time budget; continue with --resume" if stopped else "finished")
    logger.finish("budget_stopped" if stopped else "finished", **summary)
    (logger.run_dir / "val_metrics.json").write_text(json.dumps(summary.get("val"), indent=2), encoding="utf-8")
    summary["run_dir"] = str(logger.run_dir)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="section.key=value")
    parser.add_argument("--resume", help="run folder or checkpoint to continue")
    parser.add_argument("--run-dir", help="fixed run folder; continues it if it holds a checkpoint")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    run_dir = args.run_dir
    if args.resume:
        run_dir = adopt_run_dir(args.resume, resolve_output_root(cfg["run"]["output_root"]))
    summary = run_training(cfg, run_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "val"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
