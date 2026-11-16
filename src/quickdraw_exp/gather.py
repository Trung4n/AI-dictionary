"""Gather every finished run and the validation predictions into one folder: the
checkpoint package for submission, and the input of the final evaluation final_v2.
CPU only; trains nothing, predicts nothing, opens no data split.

    PYTHONPATH=src python -m quickdraw_exp.gather --inputs /kaggle/input --out /kaggle/working/package

Searches the input folders for finished runs (``<run>/checkpoints/best.pt`` with a
``summary.json`` that says finished) and for validation prediction files (``*.npz``
holding both validation halves; test predictions hold one half and are skipped).
Writes

    runs/<run>/checkpoints/best.pt  the model weights and training counters only (no optimiser)
    runs/<run>/config.yaml    the resolved configuration the run was trained with
    runs/<run>/summary.json   its training summary and final validation scores
    val_predictions/<run>.npz checkpoint predictions on the validation halves
    manifest.json             sources, sha256 of every original and written file

A run or prediction found twice must be byte-identical; otherwise the script stops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_runs(roots) -> dict:
    """{run name: run folder} of finished runs; identical copies are allowed."""
    found, digest = {}, {}
    for root in roots:
        for best in sorted(Path(root).rglob("checkpoints/best.pt")):
            run = best.parent.parent
            summary = run / "summary.json"
            if not (summary.is_file() and (run / "config.yaml").is_file()):
                continue
            if not json.loads(summary.read_text()).get("finished"):
                continue
            d = sha256(best)
            if run.name in found and digest[run.name] != d:
                raise SystemExit(f"{run.name}: two different checkpoints ({found[run.name]} and {run})")
            found.setdefault(run.name, run)
            digest.setdefault(run.name, d)
    return found


def is_validation_prediction(path: Path) -> bool:
    try:
        with np.load(path) as z:
            return "half" in z.files and set(np.unique(z["half"]).tolist()) == {0, 1}
    except Exception:
        return False


def find_val_predictions(roots) -> dict:
    found, digest = {}, {}
    for root in roots:
        for p in sorted(Path(root).rglob("*.npz")):
            if not is_validation_prediction(p):
                continue
            d = sha256(p)
            if p.stem in found and digest[p.stem] != d:
                raise SystemExit(f"{p.stem}: two different validation predictions ({found[p.stem]} and {p})")
            found.setdefault(p.stem, p)
            digest.setdefault(p.stem, d)
    return found


def slim_checkpoint(source: Path, target: Path) -> None:
    """Weights and counters only: what load_run needs, without the optimiser state."""
    import torch
    state = torch.load(source, map_location="cpu", weights_only=False)
    keep = {k: state[k] for k in ("model", "epoch", "global_step", "best_metric") if k in state}
    torch.save(keep, target)


def gather(roots, out: Path) -> dict:
    runs, preds = find_runs(roots), find_val_predictions(roots)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"runs": {}, "val_predictions": {}}
    for name, run in sorted(runs.items()):
        target = out / "runs" / name
        (target / "checkpoints").mkdir(parents=True, exist_ok=True)
        slim_checkpoint(run / "checkpoints" / "best.pt", target / "checkpoints" / "best.pt")
        for f in ("config.yaml", "summary.json"):
            shutil.copy(run / f, target / f)
        files = [target / "config.yaml", target / "summary.json", target / "checkpoints" / "best.pt"]
        manifest["runs"][name] = {"source": str(run), "best_pt_original_sha256": sha256(run / "checkpoints" / "best.pt"),
                                  "files": {str(f.relative_to(target)): sha256(f) for f in files}}
        print(f"run {name} <- {run}", flush=True)
    (out / "val_predictions").mkdir(exist_ok=True)
    for name, p in sorted(preds.items()):
        shutil.copy(p, out / "val_predictions" / p.name)
        manifest["val_predictions"][name] = {"source": str(p), "sha256": sha256(p)}
        print(f"predictions {name} <- {p}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=["/kaggle/input"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect", default=None, help="a matrix (e.g. configs/final_v2.yaml): stop if one of its runs is missing")
    args = parser.parse_args(argv)
    m = gather(args.inputs, Path(args.out))
    print(f"\n{len(m['runs'])} runs, {len(m['val_predictions'])} validation prediction files")
    if args.expect:
        import yaml

        from .config import repo_path
        spec = yaml.safe_load(repo_path(args.expect).read_text(encoding="utf-8"))
        wanted = set(spec.get("leak_runs", [])) | set(spec.get("anytime_runs", [])) | set(spec.get("hierarchy_runs", []))
        need_preds = set(spec.get("anytime_runs", [])) | set(spec.get("hierarchy_runs", []))
        missing = sorted(wanted - set(m["runs"])) + sorted(f"{r} (val predictions)" for r in need_preds - set(m["val_predictions"]))
        if missing:
            print("MISSING for", args.expect, ":", missing)
            return 1
        print("everything", args.expect, "needs is here")
    print(f"GATHER COMPLETE. Records: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
