"""Error analysis on validation, from saved checkpoint predictions. CPU only; trains
nothing and opens no test data.

    PYTHONPATH=src python -m quickdraw_exp.errors --predictions <predictions_all> [--out runs/errors_v1]

On the evaluation half of validation (thresholds, where needed, from the calibration
half, exactly as in the main results):

1. Early confusions. The answer in force when 25% (and 50%) of the ink is drawn: which
   true -> predicted pairs are most frequent, and how often a wrong answer stays inside
   the true class's semantic group, subgroup or shape cluster, against chance.
2. Stopping by class. With max-softmax stopping at a 5% accuracy gap: how much ink
   each class needs before the answer, how often its early answers are wrong, and how
   this relates to the class statistics computed before training (recognized rate,
   early confusability in reports/eda/tables/class_summary.csv).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RUNS = ["cnn_prefix_s42", "cnn_prefix_s43", "gru_uniform_s42", "gru_uniform_s43"]
AT = (0.25, 0.5)
GAP = 0.05
TOP = 15


def answer_at(ev: dict, at: float) -> tuple[np.ndarray, np.ndarray]:
    """Per drawing, the predicted class in force once ``at`` of the ink is drawn (the last
    checkpoint at or before it) and whether there is one yet."""
    starts, ends = ev["starts"], ev["ends"]
    reached = ev["ink"] <= at + 1e-12
    big = -1
    idx = np.where(reached, np.arange(len(reached)), big)
    last = np.maximum.reduceat(idx, starts)
    has = last >= starts
    pred = ev["probs"][np.where(has, last, starts)].argmax(axis=1)
    return pred, has


def group_maps(tree, n: int) -> dict:
    """Class -> index of its node at each level above the classes (or -1)."""
    out = {}
    for level in (1, 2):
        m = np.full(n, -1)
        for i in np.flatnonzero(tree.level == level):
            m[tree.members[i] > 0] = i
        out[level] = m
    return out


def within(true: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """Share of (true, pred) errors inside the true class's group, and the share a wrong
    answer drawn uniformly from the other classes would reach."""
    if len(true) == 0:
        return float("nan"), float("nan")
    same = (groups[true] >= 0) & (groups[true] == groups[pred])
    size = np.array([(groups == groups[t]).sum() if groups[t] >= 0 else 1 for t in range(len(groups))])
    chance = (size[true] - 1) / (len(groups) - 1)
    return float(same.mean()), float(chance.mean())


def early_confusions(parts: dict, names: list[str], semantic, shape, at: float) -> dict:
    """Pooled over runs of one model: accuracy, frequent pairs, and errors within groups."""
    n = len(names)
    sem, shp = group_maps(semantic, n), group_maps(shape, n)
    counts = np.zeros((n, n), np.int64)
    trues, preds, total, missing = [], [], 0, 0
    for ev in parts:
        pred, has = answer_at(ev, at)
        label = ev["label"][ev["starts"]]
        total += len(label)
        missing += int((~has).sum())
        np.add.at(counts, (label[has], pred[has]), 1)
        wrong = has & (pred != label)
        trues.append(label[wrong])
        preds.append(pred[wrong])
    true, pred = np.concatenate(trues), np.concatenate(preds)
    per_class = counts.sum(axis=1)
    off = counts.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1], off.shape))[0][:TOP]
    pairs = [{"true": names[t], "predicted": names[p], "share_of_true_class": float(off[t, p] / max(per_class[t], 1)),
              "same_semantic_group": bool(sem[2][t] >= 0 and sem[2][t] == sem[2][p]),
              "same_shape_cluster": bool(shp[2][t] >= 0 and shp[2][t] == shp[2][p])} for t, p in order]
    report = {"at": at, "drawings": total, "no_checkpoint_yet": missing,
              "accuracy": float(np.trace(counts) / max(counts.sum(), 1)), "errors": int(len(true)),
              "top_pairs": pairs}
    for label, maps in (("semantic_group", sem[2]), ("semantic_subgroup", sem[1]), ("shape_cluster", shp[2])):
        share, chance = within(true, pred, maps)
        report[f"errors_within_{label}"] = {"share": share, "chance": chance}
    return report


def stopping_by_class(parts: list[dict], names: list[str]) -> dict:
    """Max-softmax stopping at the calibrated 5% gap, per class, pooled over runs."""
    from .anytime import accuracy_gap, calibrate, halt_rows, thresholds
    n = len(names)
    ink, correct, gap_hit, labels = [], [], [], []
    for cal, ev in parts:
        s_c, s_e = cal["scores"]["pmax"], ev["scores"]["pmax"]
        tau = calibrate(s_c, cal["correct"], cal["ink"], cal["starts"], cal["ends"], thresholds(s_c), GAP)
        rows = halt_rows(s_e >= tau, ev["starts"], ev["ends"])
        ink.append(ev["ink"][rows])
        correct.append(ev["correct"][rows])
        gap_hit.append(accuracy_gap(ev["correct"], rows, ev["ends"]))
        labels.append(ev["label"][ev["starts"]])
    ink, correct, gap_hit, labels = map(np.concatenate, (ink, correct, gap_hit, labels))
    rows = []
    for c in range(n):
        m = labels == c
        rows.append({"class": names[c], "drawings": int(m.sum()), "mean_ink": float(ink[m].mean()),
                     "accuracy": float(correct[m].mean()), "early_error": float(gap_hit[m].mean())})
    return {"gap_target": GAP, "overall": {"mean_ink": float(ink.mean()), "early_error": float(gap_hit.mean())},
            "classes": rows}


def relate_to_eda(by_class: list[dict]) -> dict:
    """Spearman correlation of per-class stopping ink with statistics known before training."""
    import pandas as pd
    from scipy.stats import spearmanr

    from .config import repo_path
    table = pd.read_csv(repo_path("reports/eda/tables/class_summary.csv")).set_index("word")
    ink = pd.Series({r["class"]: r["mean_ink"] for r in by_class})
    out = {}
    for column in ("recognized_rate", "early_confusability", "n_strokes_median"):
        if column in table:
            both = pd.concat([ink, table[column]], axis=1, join="inner").dropna()
            rho, p = spearmanr(both.iloc[:, 0], both.iloc[:, 1])
            out[column] = {"spearman": float(rho), "p_value": float(p), "classes": int(len(both))}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="folder holding validation <run>.npz")
    parser.add_argument("--runs", nargs="*", default=RUNS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    import yaml

    from .analysis import _parts
    from .config import repo_path
    from .hierarchy import build_trees, class_names_from_data
    from .paths import resolve_output_root

    out = Path(args.out) if args.out else resolve_output_root(None) / "errors_v1"
    out.mkdir(parents=True, exist_ok=True)
    names = class_names_from_data()
    spec = yaml.safe_load(repo_path("configs/hierarchy_eval_v1.yaml").read_text(encoding="utf-8"))
    trees = build_trees(spec, names)
    semantic, shape = trees["semantic"][0], trees["shape"][0]

    by_model = {}
    for run in args.runs:
        path = Path(args.predictions) / f"{run}.npz"
        if not path.is_file():
            print(f"missing {run}", flush=True)
            continue
        by_model.setdefault(run.rsplit("_s", 1)[0], []).append(_parts(dict(np.load(path))))
    result = {"runs": args.runs, "models": {}}
    for model, parts in sorted(by_model.items()):
        evs = [ev for _, ev in parts]
        stopping = stopping_by_class(parts, names)
        ranked = sorted(stopping["classes"], key=lambda r: r["mean_ink"])
        result["models"][model] = {
            "seeds": len(parts),
            "early_confusions": [early_confusions(evs, names, semantic, shape, at) for at in AT],
            "stopping": stopping,
            "earliest_classes": ranked[:10], "latest_classes": ranked[-10:],
            "stopping_ink_vs_eda": relate_to_eda(stopping["classes"])}
        print(f"\n== {model} ({len(parts)} seeds)")
        for rep in result["models"][model]["early_confusions"]:
            w = rep["errors_within_semantic_group"]
            print(f"  at {rep['at']:.0%} ink: accuracy {rep['accuracy']:.3f}, errors within the semantic group "
                  f"{w['share']:.3f} (chance {w['chance']:.3f})")
            for p in rep["top_pairs"][:8]:
                print(f"    {p['true']} -> {p['predicted']}: {p['share_of_true_class']:.3f}"
                      f"{'  same group' if p['same_semantic_group'] else ''}")
        print(f"  stopping at 5% gap: mean ink {stopping['overall']['mean_ink']:.3f}")
        print("  earliest:", [(r["class"], round(r["mean_ink"], 2)) for r in ranked[:5]])
        print("  latest:  ", [(r["class"], round(r["mean_ink"], 2)) for r in ranked[-5:]])
        print("  vs EDA:", result["models"][model]["stopping_ink_vs_eda"])
    (out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nERROR ANALYSIS COMPLETE. Records: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
