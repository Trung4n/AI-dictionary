"""Find the processed dataset and the output folder on any machine.

Search order for the data: ``data.root`` in the config, the ``QUICKDRAW_DATA``
environment variable, ``dataset/processed``, ``dataset``, then ``/kaggle/input``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .config import REPO_ROOT

SPLITS = ("train", "val", "test")
SPLIT_FILES = ("xy.npy", "end.npy", "offsets.npy", "meta.parquet")
REQUIRED_FILES = ("manifest.json", "seq_stats.json",
                  *(f"{s}_{k}" for s in SPLITS for k in SPLIT_FILES))
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def _find_manifests(base: Path, max_depth: int = 5) -> list[Path]:
    """Folders under ``base`` holding a manifest.json. Kaggle's mount layout
    varies, so the dataset's slug and depth are not assumed."""
    found = []
    base_depth = len(base.parts)
    for folder, subdirs, files in os.walk(base):
        if "manifest.json" in files:
            found.append(Path(folder))
            subdirs[:] = []
        elif len(Path(folder).parts) - base_depth >= max_depth:
            subdirs[:] = []
    return sorted(found)


def missing_files(root: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (root / name).is_file()]


def resolve_data_root(explicit=None, kaggle_input: Path = KAGGLE_INPUT) -> Path:
    named = explicit or os.environ.get("QUICKDRAW_DATA")
    if named:
        root = Path(named)
        missing = missing_files(root)
        if missing:
            raise FileNotFoundError(f"{root} is missing: {', '.join(missing)}")
        return root
    for root in (REPO_ROOT / "dataset" / "processed", REPO_ROOT / "dataset"):
        if not missing_files(root):
            return root
    matches = [m for m in _find_manifests(kaggle_input) if not missing_files(m)] if kaggle_input.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(f"several datasets found, set data.root to one: {matches}")
    raise FileNotFoundError(
        "processed dataset not found. Set data.root or QUICKDRAW_DATA, or attach the "
        f"Kaggle Dataset holding these files: {', '.join(REQUIRED_FILES)}")


def resolve_output_root(explicit=None, kaggle_working: Path = KAGGLE_WORKING) -> Path:
    if explicit:
        return Path(explicit)
    return kaggle_working / "runs" if kaggle_working.is_dir() else REPO_ROOT / "runs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data(root: Path, split_version: str, split_digest: str, check_sha256: bool = False) -> dict:
    """Stop if this is not the agreed split. Returns the manifest."""
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["split_version"] != split_version:
        raise ValueError(f"split version {manifest['split_version']} != {split_version}")
    if manifest["split_digest"] != split_digest:
        raise ValueError("split_digest differs from the config: this is a different split")
    if check_sha256:
        bad = [name for name, want in manifest["files"].items() if _sha256(root / name) != want]
        if bad:
            raise ValueError(f"files differ from manifest.json: {', '.join(bad)}")
    return manifest
