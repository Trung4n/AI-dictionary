"""Final evaluation (configs/final_v1.yaml). Trains nothing; thresholds come from validation.

    PYTHONPATH=src python -m quickdraw_exp.final_eval --split val  --val-predictions <folder of val .npz>
    PYTHONPATH=src python -m quickdraw_exp.final_eval --split test --val-predictions <folder of val .npz>

``--split val`` repeats the whole pipeline on validation data and checks it against
the reported numbers (``reports/`` and the saved validation predictions); it exits
with an error when they disagree. ``--split test`` opens the test split and the
held-out countries, once, and writes the final results.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from .anytime import (aggregate, build_items, calibration_split, compare, condition_of, evaluate_policies,
                      load_run, predict_items)

TOLERANCE = 0.002  # largest accepted difference from a reported validation number


# ---------------------------------------------------------------- pieces without models

def sample_positions(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Up to ``per_class`` positions of each class, sorted, fixed by ``seed``."""
    rng = np.random.default_rng(seed)
    picked = [rng.permutation(np.flatnonzero(labels == c))[:per_class] for c in np.unique(labels)]
    return np.sort(np.concatenate(picked))


def with_val_calibration(val_saved: dict, evaluated: dict) -> dict:
    """Validation calibration half (half 0) followed by the rows to evaluate (half 1).
    Drawing ids of the evaluated rows are shifted so the two parts never share one."""
    calib = val_saved["half"] == 0
    shift = int(val_saved["drawing"].max()) + 1
    out = {}
    for key in ("probs", "confidence", "label", "ink", "strokes", "drawing"):
        b = evaluated[key] + shift if key == "drawing" else evaluated[key]
        out[key] = np.concatenate([val_saved[key][calib], b])
    out["half"] = np.concatenate([np.zeros(int(calib.sum()), np.int8), np.ones(len(evaluated["label"]), np.int8)])
    return out


def early_top1(by_frac: dict, fracs) -> float:
    return float(np.mean([by_frac[str(f)]["top1"] for f in fracs]))


def leak_comparisons(per_run: dict, comparisons: list[dict], fracs) -> list[dict]:
    rows = []
    for c in comparisons:
        a = [early_top1(v, fracs) for r, v in per_run.items() if condition_of(Path(r))[0] == c["a"]]
        b = [early_top1(v, fracs) for r, v in per_run.items() if condition_of(Path(r))[0] == c["b"]]
        row = {"name": c["name"]}
        if a and b:
            row.update({"a_mean": float(np.mean(a)), "b_mean": float(np.mean(b)), "difference": float(np.mean(a) - np.mean(b)),
                        "a_values": a, "b_values": b})
        else:
            row["difference"] = None
        rows.append(row)
    return rows


def find_run_dirs(names, roots) -> dict:
    """``<name>/checkpoints/best.pt`` of a finished run, searched under ``roots``."""
    found = {}
    for root in roots:
        for best in sorted(Path(root).rglob("checkpoints/best.pt")):
            run = best.parent.parent
            if run.name in names and run.name not in found and (run / "summary.json").is_file():
                if json.loads((run / "summary.json").read_text()).get("finished"):
                    found[run.name] = run
    return found


def reported_top1(paths) -> dict:
    """``{run: {frac: top1}}`` from the experiments' results.csv files."""
    out = {}
    for path in paths:
        if Path(path).is_file():
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    out.setdefault(row["job"], {})[row["frac"]] = float(row["top1"])
    return out


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--matrix", default="configs/final_v1.yaml")
    parser.add_argument("--val-predictions", required=True, help="folder with the saved validation <run>.npz")
    parser.add_argument("--runs", nargs="*", default=["/kaggle/input"], help="folders searched for checkpoints")
    parser.add_argument("--out", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", type=int, default=None,
                        help="dry run only: N drawings per class everywhere (validation predictions saved with per_class=N)")
    args = parser.parse_args(argv)

    import torch

    from .config import repo_path
    from .data import load_holdout, open_split, open_unseen_countries
    from .engine import evaluate, pick_device, stratified_subset
    from .hierarchy import build_trees, class_names_from_data, run_trees
    from .hierarchy import compare as hierarchy_compare
    from .paths import resolve_data_root, resolve_output_root

    spec = yaml.safe_load(repo_path(args.matrix).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else resolve_output_root(None) / f"{spec['experiment']}_{args.split}{'_smoke' if args.smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device("auto")
    root = resolve_data_root(None)
    base = yaml.safe_load(repo_path("configs/base.yaml").read_text(encoding="utf-8"))
    holdout = load_holdout(repo_path(base["data"]["holdout_file"]))

    names = set(spec["leak_runs"]) | set(spec["anytime_runs"]) | set(spec["hierarchy_runs"]) | set(spec["unseen_runs"])
    runs = find_run_dirs(names, args.runs)
    missing = sorted(names - set(runs))
    print("checkpoints:", {k: str(v) for k, v in sorted(runs.items())}, flush=True)
    if missing:
        raise SystemExit(f"checkpoints not found: {missing}")
    val_folder = Path(args.val_predictions)
    val_saved = {r: dict(np.load(val_folder / f"{r}.npz")) for r in set(spec["anytime_runs"]) | set(spec["hierarchy_runs"])}

    # The sets evaluated: validation (the check) or test plus the held-out countries.
    val = open_split("val", root, holdout)
    if args.split == "val":
        evals = calibration_split(np.asarray(val.labels), *([args.smoke] if args.smoke else []))[1]
        sets = {"val": {"split": val, "positions": evals, "leak": "per_run_final", "runs": names}}
    else:
        test = open_split("test", root, holdout, final_evaluation=True)
        unseen = open_unseen_countries(root, holdout, final_evaluation=True)
        per_class = args.smoke or spec["per_class"]
        sets = {"test": {"split": test, "positions": sample_positions(np.asarray(test.labels), per_class, spec["sample_seed"]),
                         "leak": "all", "runs": names},
                "unseen_countries": {"split": unseen, "positions": sample_positions(np.asarray(unseen.labels), per_class, spec["sample_seed"]),
                                     "leak": "all", "runs": set(spec["unseen_runs"])}}

    spec_h = yaml.safe_load(repo_path(spec["hierarchy_matrix"]).read_text(encoding="utf-8"))
    trees = build_trees(spec_h, class_names_from_data())
    grid = spec_h["tau_grid"]
    taus = np.round(np.arange(grid["start"], grid["stop"] + 1e-9, grid["step"]), 4)
    targets = spec_h["precision_targets"]

    result = {"experiment": spec["experiment"], "split": args.split, "torch": torch.__version__,
              "checkpoints": {k: str(v) for k, v in sorted(runs.items())}, "sets": {}}
    check = {}
    for set_name, s in sets.items():
        split, wanted = s["split"], s["runs"]
        entry = {"drawings": len(split), "sampled_drawings": int(len(s["positions"]))}
        print(f"\n=== {set_name}: {len(split)} drawings, {len(s['positions'])} sampled for checkpoints", flush=True)

        # Claim 1: top-1 by ink fraction.
        leak = {}
        for name in spec["leak_runs"]:
            if name not in wanted:
                continue
            cfg, model = load_run(runs[name], device)
            cfg["data"]["num_workers"] = args.num_workers
            target = split
            if s["leak"] == "per_run_final":
                target = stratified_subset(split, cfg["eval"]["final_per_class"], cfg["run"]["seed"])
            if args.smoke:
                target = stratified_subset(split, args.smoke, cfg["run"]["seed"])
            scores = evaluate(model, cfg, target, device)
            leak[name] = {f: {k: v[k] for k in ("n", "top1", "top3", "macro_f1", "ece")} for f, v in scores["by_frac"].items()}
            print(f"top-1 {name}: " + ", ".join(f"{f} {v['top1']:.4f}" for f, v in leak[name].items()), flush=True)
            del model
        entry["leak"] = {"per_run": leak, "comparisons": leak_comparisons(leak, spec["leak_comparisons"], spec["early_fracs"])}

        # Claims 2 and 3: checkpoint predictions, thresholds from the validation calibration half.
        items = build_items(split, s["positions"])
        items["half"] = np.ones(len(items["k"]), np.int8)
        per_run, hier, preds = {}, {}, {}
        for name in sorted((set(spec["anytime_runs"]) | set(spec["hierarchy_runs"])) & wanted):
            cfg, model = load_run(runs[name], device)
            pred = predict_items(model, cfg, split, items, device, root, args.num_workers)
            del model
            if args.split == "val":
                preds[name] = pred
            else:  # kept for later analyses of the final results
                (out / f"predictions_{set_name}").mkdir(exist_ok=True)
                np.savez_compressed(out / f"predictions_{set_name}" / f"{name}.npz", **pred)
            data = with_val_calibration(val_saved[name], pred)
            if name in spec["anytime_runs"]:
                per_run[condition_of(Path(name))] = evaluate_policies(data)
            if name in spec["hierarchy_runs"]:
                hier[name] = run_trees(data, trees, taus, targets)
            print(f"scored {name}", flush=True)
        table = aggregate(per_run)
        entry["anytime"] = {"comparisons": compare(table, spec["anytime_comparisons"]),
                            "per_run": {f"{c}_s{sd}": {p: {"auc": r["auc"], "risk_control": r["risk_control"]}
                                                        for p, r in rep.items() if p not in ("reference", "fixed_strokes")}
                                        | {"reference": rep["reference"]}
                                        for (c, sd), rep in per_run.items()}}
        entry["hierarchy"] = {"comparisons": [row for t in targets for row in hierarchy_compare(hier, spec["hierarchy_comparisons"], str(t))],
                              "per_run": hier}
        result["sets"][set_name] = entry

        if args.split == "val":
            # Same numbers as reported? Top-1 against the results.csv files, the rest against the saved predictions.
            reported = reported_top1([repo_path(p) for p in ("reports/leak_v1/results.csv", "reports/followup_v1/results.csv")])
            diffs = {}
            for name, by_frac in leak.items():
                for f, v in by_frac.items():
                    if name in reported and f in reported[name]:
                        diffs[f"top1 {name} {f}"] = abs(v["top1"] - reported[name][f])
            for name, pred in preds.items():
                saved = val_saved[name]
                ev = saved["half"] == 1
                diffs[f"probs {name}"] = float(np.abs(saved["probs"][ev].astype(np.float32) - pred["probs"].astype(np.float32)).max())
                old = evaluate_policies(saved)
                new = evaluate_policies(with_val_calibration(saved, pred))
                for p in ("pmax", "conf"):
                    if p in old and p in new:
                        diffs[f"auc {name} {p}"] = abs(old[p]["auc"] - new[p]["auc"])
                if name in hier:
                    old_h = run_trees(saved, trees, taus, targets)
                    for t in map(str, targets):
                        for tree in ("semantic", "flat"):
                            diffs[f"hierarchy {name} {t} {tree}"] = abs(old_h[t][tree]["reward_area"] - hier[name][t][tree]["reward_area"])
            worst = max(diffs, key=diffs.get)
            check = {"tolerance": TOLERANCE, "compared": len(diffs), "largest": {worst: diffs[worst]},
                     "failed": {k: v for k, v in diffs.items() if v > TOLERANCE and not k.startswith("probs")},
                     "probs_max_abs_diff": max(v for k, v in diffs.items() if k.startswith("probs"))}
            result["check"] = check

    (out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if args.split == "val":
        ok = not check["failed"]
        print(f"\nVALIDATION CHECK {'PASSED' if ok else 'FAILED'}: {check['compared']} numbers, largest {check['largest']}, "
              f"failed {check['failed']}")
        print(f"Records: {out}")
        return 0 if ok else 1
    print(f"\nFINAL EVALUATION COMPLETE. Records: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
