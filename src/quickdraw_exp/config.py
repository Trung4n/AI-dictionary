"""YAML configuration with ``extends`` and command-line overrides."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict, update: dict) -> dict:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"config extends itself: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return _merge(_load(path.parent / parent, (*seen, path)), data)


def apply_override(cfg: dict, item: str) -> None:
    """Apply ``section.key=value``. The key must already exist, so a typo fails
    instead of silently running the wrong experiment."""
    if "=" not in item:
        raise ValueError(f"override must look like section.key=value, got {item!r}")
    dotted, raw = item.split("=", 1)
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            raise KeyError(f"unknown config section {dotted!r}")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"unknown config key {dotted!r}")
    node[keys[-1]] = yaml.safe_load(raw)


def load_config(path, overrides=()) -> dict:
    cfg = _load(Path(path))
    for item in overrides:
        apply_override(cfg, item)
    return cfg


def repo_path(path) -> Path:
    """A path from a config: absolute as given, otherwise relative to the repo."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
