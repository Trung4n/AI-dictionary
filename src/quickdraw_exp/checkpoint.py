"""Save everything needed to continue a run exactly where it stopped."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, *, epoch: int, batches_done: int, global_step: int,
                    scheduler=None, scaler=None, best_metric=None) -> Path:
    """Write to a temporary file, then rename: a session that dies while saving
    leaves the previous checkpoint intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "batches_done": batches_done,
        "global_step": global_step,
        "best_metric": best_metric,
        "rng": capture_rng(),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None) -> dict:
    """Restore state and return the counters. Move the model to its device first.
    The file holds Python objects, so only load checkpoints this project made."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return {key: state[key] for key in ("epoch", "batches_done", "global_step", "best_metric")}


class TimeBudget:
    """Stop before a session or the weekly quota runs out, with time left to save."""

    def __init__(self, max_minutes: float | None, clock=time.monotonic):
        self.clock, self.start = clock, clock()
        self.max_seconds = None if max_minutes is None else 60.0 * max_minutes

    def exhausted(self) -> bool:
        return self.max_seconds is not None and self.clock() - self.start >= self.max_seconds
