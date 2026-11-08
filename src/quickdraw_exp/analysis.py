"""Analyses after the experiments, from saved checkpoint predictions. CPU only;
trains nothing and opens no test data.

    PYTHONPATH=src python -m quickdraw_exp.analysis --predictions <folder of <run>.npz> [--out runs/analysis_v1]

1. By ``recognized``. The game stops a round once Google's model recognises the
   drawing, so for recognised drawings 100% ink means "the round stopped", not
   "the drawing was finished" (reports/preprocessing/leak_audit.md). Every headline
   comparison is repeated on recognised and unrecognised drawings separately.
   Thresholds still come from the whole calibration half, as in the main results;
   only the evaluation half is split.
2. Hierarchy by group. Where the semantic tree's gain over flat answers comes
   from: per top-level group of the frozen tree, and the classes that gain most
   and least. Same rules as configs/hierarchy_eval_v1.yaml (precision 0.95).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from .anytime import group_bounds, policy_report, scores_from

# Comparisons repeated per subgroup: (name, (run condition, policy), (run condition, policy)).
COMPARISONS = [
    ("residual leak: uniform vs prefix, GRU", ("gru_uniform", "pmax"), ("gru_prefix", "pmax")),
    ("method vs baseline, GRU uniform", ("gru_anytime_uniform", "conf"), ("gru_uniform", "pmax")),
    ("method vs baseline, GRU prefix", ("gru_anytime", "conf"), ("gru_prefix", "pmax")),
    ("method vs baseline, CNN", ("cnn_anytime", "conf"), ("cnn_prefix", "pmax")),
    ("same compute 2x: method vs baseline, GRU uniform", ("gru_anytime_uniform_long", "conf"), ("gru_uniform_long", "pmax")),
]
HIERARCHY_RUNS = ["gru_uniform_s42", "gru_uniform_s43", "gru_prefix_s42", "gru_prefix_s43", "cnn_prefix_s42", "cnn_prefix_s43"]
TARGET = 0.95


def _parts(data: dict, eval_keep: np.ndarray | None = None) -> tuple[dict, dict]:
    """Calibration half (whole) and evaluation half (optionally only some drawings)."""
    out = []
    for flag in (0, 1):
        mask = data["half"] == flag
        if flag == 1 and eval_keep is not None:
            mask &= eval_keep[data["drawing"]]
        part = {key: data[key][mask] for key in ("probs", "confidence", "label", "ink", "strokes", "drawing")}
        part["correct"] = (part["probs"].argmax(axis=1) == part["label"]).astype(np.float64)
        part["starts"], part["ends"] = group_bounds(part["drawing"])
        part["scores"] = scores_from(part["probs"], part["confidence"])
        out.append(part)
    return out[0], out[1]


def policies_on(data: dict, eval_keep: np.ndarray | None = None) -> dict:
    """AUC, risk control and end accuracy per policy, on the kept evaluation drawings."""
    cal, ev = _parts(data, eval_keep)
    report = {}
    for name in ev["scores"]:
        r = policy_report(cal["scores"][name], ev["scores"][name], cal, ev)
        report[name] = {"auc": r["auc"], "risk_control": {g: {k: v[k] for k in ("gap", "met", "earliness")}
                                                          for g, v in r["risk_control"].items()}}
    report["end_accuracy"] = float(ev["correct"][ev["ends"] - 1].mean())
    report["drawings"] = int(len(ev["starts"]))
    return report


def by_recognized(predictions: dict, recognized: np.ndarray) -> dict:
    """Per run and subgroup (all, recognized, unrecognized), plus the comparisons."""
    groups = {"all": None, "recognized": recognized, "unrecognized": ~recognized}
    per_run = {run: {g: policies_on(data, keep) for g, keep in groups.items()} for run, data in predictions.items()}
    rows = []
    for name, (cond_a, pol_a), (cond_b, pol_b) in COMPARISONS:
        runs_a = sorted(r for r in per_run if r.rsplit("_s", 1)[0] == cond_a)
        runs_b = sorted(r for r in per_run if r.rsplit("_s", 1)[0] == cond_b)
        if not runs_a or not runs_b:
            rows.append({"name": name, "missing": True})
            continue
        row = {"name": name, "a": [cond_a, pol_a], "b": [cond_b, pol_b]}
        for g in groups:
            a = float(np.mean([per_run[r][g][pol_a]["auc"] for r in runs_a]))
            b = float(np.mean([per_run[r][g][pol_b]["auc"] for r in runs_b]))
            row[g] = {"a": a, "b": b, "difference": a - b}
        rows.append(row)
    return {"per_run": per_run, "comparisons": rows}


def hierarchy_by_group(data: dict, semantic, flat, taus, class_names: list[str]) -> dict:
    """Per-drawing reward area for the semantic tree and flat answers, then by group and class."""
    from .hierarchy import answer, calibrate, halves, reward_area, score
    parts = halves(data)
    cal, ev = parts["calib"], parts["eval"]
    areas, level_share = {}, {}
    for name, tree in (("semantic", semantic), ("flat", flat)):
        tau = calibrate(cal["probs"], cal["label"], tree, taus, TARGET)
        nodes = answer(ev["probs"] @ tree.members.T, tree, tau)
        _, reward = score(nodes, ev["label"], tree)
        areas[name] = reward_area(reward, ev["ink"], ev["starts"], ev["ends"])
        if name == "semantic":
            levels = np.where(nodes >= 0, tree.level[np.where(nodes >= 0, nodes, 0)], -1)
    label = ev["label"][ev["starts"]]
    gain = areas["semantic"] - areas["flat"]
    group_of = {}
    for i, lvl in enumerate(semantic.level):
        if lvl == 2:
            for c in np.flatnonzero(semantic.members[i]):
                group_of[int(c)] = semantic.names[i]
    by_group = {}
    for group in sorted(set(group_of.values())):
        classes = [c for c, g in group_of.items() if g == group]
        drawings = np.isin(label, classes)
        rows = np.isin(ev["label"], classes)
        by_group[group] = {"classes": len(classes), "drawings": int(drawings.sum()),
                           "semantic": float(areas["semantic"][drawings].mean()),
                           "flat": float(areas["flat"][drawings].mean()),
                           "gain": float(gain[drawings].mean()),
                           "share_of_total_gain": float(gain[drawings].sum() / gain.sum()) if gain.sum() else 0.0,
                           "answers_at_subgroup": float((levels[rows] == 1).mean()),
                           "answers_at_group": float((levels[rows] == 2).mean())}
    per_class = {class_names[c]: float(gain[label == c].mean()) for c in np.unique(label)}
    ranked = sorted(per_class.items(), key=lambda kv: kv[1], reverse=True)
    return {"gain": float(gain.mean()), "by_group": by_group, "top_classes": ranked[:10], "bottom_classes": ranked[-10:]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="folder holding <run>.npz")
    parser.add_argument("--out", default=None, help="default: runs/analysis_v1 (or /kaggle/working/runs/...)")
    args = parser.parse_args(argv)

    from .config import repo_path
    from .data import load_holdout, open_split
    from .hierarchy import build_trees, class_names_from_data
    from .paths import resolve_data_root, resolve_output_root

    out = Path(args.out) if args.out else resolve_output_root(None) / "analysis_v1"
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(repo_path("configs/base.yaml").read_text(encoding="utf-8"))
    val = open_split("val", resolve_data_root(None), load_holdout(repo_path(base["data"]["holdout_file"])))
    recognized = val.meta["recognized"].to_numpy().astype(bool)

    folder = Path(args.predictions)
    wanted = {c for _, a, b in COMPARISONS for c in (a[0], b[0])}
    predictions = {p.stem: dict(np.load(p)) for p in sorted(folder.glob("*.npz")) if p.stem.rsplit("_s", 1)[0] in wanted}
    print("runs:", sorted(predictions), flush=True)
    result = {"recognized_share_val": float(recognized.mean()), "by_recognized": by_recognized(predictions, recognized)}

    spec = yaml.safe_load(repo_path("configs/hierarchy_eval_v1.yaml").read_text(encoding="utf-8"))
    names = class_names_from_data()
    trees = build_trees(spec, names)
    grid = spec["tau_grid"]
    taus = np.round(np.arange(grid["start"], grid["stop"] + 1e-9, grid["step"]), 4)
    result["hierarchy_by_group"] = {}
    for run in HIERARCHY_RUNS:
        path = folder / f"{run}.npz"
        if path.is_file():
            data = predictions.get(run) or dict(np.load(path))
            result["hierarchy_by_group"][run] = hierarchy_by_group(data, trees["semantic"][0], trees["flat"][0], taus, names)
            print(f"hierarchy by group: {run}", flush=True)

    (out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nANALYSIS COMPLETE. Records: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
