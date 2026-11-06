"""Experiment records: one folder per run, one line per run in index.csv.

    runs/<time>_<name>_s<seed>/
        config.yaml     resolved configuration
        env.json        software, hardware, git commit, data digest
        metrics.jsonl   one JSON line per logged step
        events.log      starts, resumes, notes
        summary.json    written when the run ends
        checkpoints/
"""
from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import REPO_ROOT

INDEX_COLUMNS = ("finished_utc", "run", "name", "seed", "account", "status", "git_commit", "summary")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _version(module: str) -> str | None:
    try:
        return __import__(module).__version__
    except ImportError:
        return None


def collect_env() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "opencv": _version("cv2"),
        "torch": _version("torch"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")),
        "gpus": [],
    }
    try:
        import torch
        env["cuda"] = torch.version.cuda
        env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return env


def new_run_dir(output_root, name: str, seed: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / f"{stamp}_{name}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def adopt_run_dir(source, output_root) -> Path:
    """Where to continue a run. A run in a read-only place (a previous Kaggle
    session attached as input) is copied into the output folder first."""
    source = Path(source)
    if source.is_file():
        source = source.parent.parent  # <run>/checkpoints/latest.pt
    if not (source / "config.yaml").is_file():
        raise FileNotFoundError(f"{source} is not a run folder")
    if os.access(source, os.W_OK) and not str(source).startswith("/kaggle/input"):
        return source
    target = Path(output_root) / source.name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    return target


class RunLogger:
    def __init__(self, run_dir, config: dict, data_info: dict | None = None):
        self.run_dir = Path(run_dir)
        self.config = config
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._started = time.time()
        config_path = self.run_dir / "config.yaml"
        if not config_path.exists():
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            env = collect_env()
            env["started_utc"] = _utc_now()
            env["account"] = config.get("run", {}).get("account")
            env["data"] = data_info or {}
            (self.run_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    def event(self, message: str) -> None:
        line = f"{_utc_now()} {message}"
        print(line, flush=True)
        with open(self.run_dir / "events.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log(self, step: int, **metrics) -> None:
        record = {"step": step, "t_s": round(time.time() - self._started, 3), **metrics}
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def rewind(self, step: int) -> int:
        """Drop metric lines after ``step``: they came from work lost after the
        last checkpoint and will be redone. Returns how many were dropped."""
        if not self.metrics_path.exists():
            return 0
        lines = self.metrics_path.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if json.loads(line)["step"] <= step]
        tmp = self.metrics_path.with_suffix(".tmp")
        tmp.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        os.replace(tmp, self.metrics_path)
        return len(lines) - len(kept)

    def read_metrics(self) -> list[dict]:
        if not self.metrics_path.exists():
            return []
        return [json.loads(line) for line in self.metrics_path.read_text(encoding="utf-8").splitlines()]

    def finish(self, status: str, **summary) -> None:
        record = {"status": status, "finished_utc": _utc_now(), **summary}
        (self.run_dir / "summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        index = self.run_dir.parent / "index.csv"
        is_new = not index.exists()
        run = self.config.get("run", {})
        with open(index, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(INDEX_COLUMNS)
            writer.writerow([record["finished_utc"], self.run_dir.name, run.get("name"), run.get("seed"),
                             run.get("account"), status, _git("rev-parse", "HEAD"), json.dumps(summary)])
