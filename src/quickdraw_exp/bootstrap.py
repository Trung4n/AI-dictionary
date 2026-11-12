"""Bootstrap intervals for the final results (final_v1). CPU only; trains nothing,
predicts nothing and opens no data split.

    PYTHONPATH=src python -m quickdraw_exp.bootstrap --val-predictions <predictions_all> \
        --final-predictions <folder holding predictions_test/ and predictions_unseen_countries/> [--out ...]

It reads the checkpoint predictions that final_eval saved for later analyses, and the
validation calibration halves, and rebuilds every reported number exactly as
final_v1 did: thresholds and tree thresholds come from validation, once. Only the
evaluated drawings are then resampled with replacement, B times, with the same draws
for every run of a set, so differences between runs are paired. Nothing is chosen
again on the resampled data. Intervals are percentile intervals (2.5%, 97.5%).

The point estimate of each quantity must equal the number in
reports/final_v1_test/results.json; the script stops if one differs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from .anytime import GAPS, GRID, accuracy_gap, calibrate, halt_rows, thresholds

B = 1000
SEED = 0
CHUNK = 50
CHECK = 1e-6


# ---------------------------------------------------------------- per-drawing quantities

def stopping_tables(data: dict) -> dict:
    """Per stopping score: for every threshold, each evaluated drawing's correctness and
    ink at the moment it answers (drawings x thresholds), and its accuracy-gap
    indicator at the risk-controlled thresholds."""
    from .analysis import _parts
    cal, ev = _parts(data)
    out = {"end_correct": ev["correct"][ev["ends"] - 1].astype(np.float32), "label": ev["label"][ev["starts"]]}
    for name in ev["scores"]:
        taus = thresholds(cal["scores"][name])
        correct = np.empty((len(ev["starts"]), len(taus)), np.float32)  # 0/1: exact sums in float32
        ink = np.empty(correct.shape, np.float64)  # float64, so frontier points match anytime.area exactly
        for j, tau in enumerate(taus):
            rows = halt_rows(ev["scores"][name] >= tau, ev["starts"], ev["ends"])
            correct[:, j], ink[:, j] = ev["correct"][rows], ev["ink"][rows]
        gaps = {}
        for gap in GAPS:
            tau = calibrate(cal["scores"][name], cal["correct"], cal["ink"], cal["starts"], cal["ends"], taus, gap)
            rows = halt_rows(ev["scores"][name] >= tau, ev["starts"], ev["ends"])
            gaps[str(gap)] = accuracy_gap(ev["correct"], rows, ev["ends"]).astype(np.float32)
        out[name] = {"correct": correct, "ink": ink, "gap": gaps}
    return out


def tree_areas(data: dict, trees: dict, taus, targets) -> dict:
    """Per target and tree: each evaluated drawing's reward area (mean over the trees
    of a kind, e.g. the five shuffles), thresholds from the calibration half."""
    from .hierarchy import answer, calibrate as calibrate_tree, halves, reward_area, score
    parts = halves(data)
    cal, ev = parts["calib"], parts["eval"]
    out = {}
    for target in targets:
        for name, tree_list in trees.items():
            areas = []
            for tree in tree_list:
                tau = calibrate_tree(cal["probs"], cal["label"], tree, taus, target)
                nodes = answer(ev["probs"] @ tree.members.T, tree, tau)
                _, reward = score(nodes, ev["label"], tree)
                areas.append(reward_area(reward, ev["ink"], ev["starts"], ev["ends"]))
            out.setdefault(str(target), {})[name] = np.mean(areas, axis=0)
    return out


# ---------------------------------------------------------------- resampling

def resample_weights(n: int, b: int, seed: int):
    """Chunks of bootstrap counts (each row sums to n), the same for every caller with the same seed."""
    rng = np.random.default_rng(seed)
    done = 0
    while done < b:
        k = min(CHUNK, b - done)
        idx = rng.integers(0, n, size=(k, n))
        w = np.zeros((k, n), np.float32)
        for r in range(k):
            w[r] = np.bincount(idx[r], minlength=n)
        done += k
        yield w


def auc_rows(earliness: np.ndarray, accuracy: np.ndarray, grid=GRID) -> np.ndarray:
    """Area under the accuracy-earliness frontier for each row (as anytime.area)."""
    ok = earliness[:, :, None] <= grid[None, None, :] + 1e-12
    return np.where(ok, accuracy[:, :, None], 0.0).max(axis=1).mean(axis=1)


def full_weights(n: int) -> np.ndarray:
    return np.ones((1, n), np.float32)


def statistics(tables: dict, hier: dict, w: np.ndarray) -> dict:
    """Every quantity of one run under counts ``w`` (rows = resamples)."""
    n = w.shape[1]
    out = {"end_accuracy": (w @ tables["end_correct"]).astype(np.float64) / n}
    for name, t in tables.items():
        if not isinstance(t, dict):
            continue
        acc = (w @ t["correct"]).astype(np.float64) / n
        ear = w.astype(np.float64) @ t["ink"] / n
        out[f"auc/{name}"] = auc_rows(ear, acc)
        for gap, g in t["gap"].items():
            out[f"gap/{name}/{gap}"] = (w @ g).astype(np.float64) / n
    for target, by_tree in hier.items():
        for tree, area in by_tree.items():
            out[f"reward/{target}/{tree}"] = w.astype(np.float64) @ area / n
    return out


def interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def condition(run: str) -> str:
    return run.rsplit("_s", 1)[0]


def run_set(per_run: dict, b: int, seed: int) -> tuple[dict, dict]:
    """Point values and bootstrap draws of every quantity, for every run of one set."""
    n = {len(t["end_correct"]) for t, _ in per_run.values()}
    if len(n) != 1:
        raise ValueError("runs of one set must share the evaluated drawings")
    n = n.pop()
    labels = [t["label"] for t, _ in per_run.values()]
    if any(not np.array_equal(labels[0], x) for x in labels[1:]):
        raise ValueError("runs of one set must evaluate the same drawings in the same order")
    point = {run: {k: float(v[0]) for k, v in statistics(t, h, full_weights(n)).items()} for run, (t, h) in per_run.items()}
    draws = {run: {} for run in per_run}
    for w in resample_weights(n, b, seed):
        for run, (t, h) in per_run.items():
            for k, v in statistics(t, h, w).items():
                draws[run].setdefault(k, []).append(v)
    draws = {run: {k: np.concatenate(v) for k, v in d.items()} for run, d in draws.items()}
    return point, draws


def seed_mean(values: dict, cond: str, key: str):
    runs = sorted(r for r in values if condition(r) == cond and key in values[r])
    if not runs:
        return None
    return np.mean([values[r][key] for r in runs], axis=0)


def compare_rows(point: dict, draws: dict, rows: list[tuple[str, str, str]]) -> list[dict]:
    """(name, key of A as condition|quantity, key of B) -> difference with its interval."""
    out = []
    for name, a, b in rows:
        (ca, ka), (cb, kb) = a.split("|"), b.split("|")
        pa, pb = seed_mean(point, ca, ka), seed_mean(point, cb, kb)
        if pa is None or pb is None:
            out.append({"name": name, "difference": None})
            continue
        diff = seed_mean(draws, ca, ka) - seed_mean(draws, cb, kb)
        out.append({"name": name, "a": float(pa), "b": float(pb), "difference": float(pa - pb), "ci95": interval(diff),
                    "share_of_draws_above_0": float((diff > 0).mean())})
    return out


# ---------------------------------------------------------------- the final_v1 numbers

def anytime_rows(spec: dict) -> list[tuple[str, str, str]]:
    return [(c["name"], f"{c['a'][0]}|auc/{c['a'][1]}", f"{c['b'][0]}|auc/{c['b'][1]}") for c in spec["anytime_comparisons"]]


def hierarchy_rows(spec: dict, targets) -> list[tuple[str, str, str]]:
    rows = []
    for t in map(str, targets):
        for c in spec["hierarchy_comparisons"]:
            cond = {condition(r) for r in c["runs"]}
            if len(cond) != 1:
                raise ValueError(f"{c['name']}: runs of more than one condition")
            cond = cond.pop()
            rows.append((f"{c['name']} @ {t}", f"{cond}|reward/{t}/{c['a']}", f"{cond}|reward/{t}/{c['b']}"))
    return rows


def check_against_report(point: dict, reported: dict) -> float:
    """Largest difference between a rebuilt point value and the reported one."""
    worst = 0.0
    for run, rep in reported["anytime"]["per_run"].items():
        for policy, r in rep.items():
            if policy == "reference":
                worst = max(worst, abs(point[run]["end_accuracy"] - r["end_of_drawing"]["accuracy"]))
                continue
            worst = max(worst, abs(point[run][f"auc/{policy}"] - r["auc"]))
            for gap, rc in r["risk_control"].items():
                worst = max(worst, abs(point[run][f"gap/{policy}/{gap}"] - rc["gap"]))
    for run, rep in reported["hierarchy"]["per_run"].items():
        for target, trees in rep.items():
            for tree, r in trees.items():
                worst = max(worst, abs(point[run][f"reward/{target}/{tree}"] - r["reward_area"]))
    return worst


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-predictions", required=True, help="folder with the validation <run>.npz (predictions_all)")
    parser.add_argument("--final-predictions", required=True,
                        help="folder holding predictions_test/ and predictions_unseen_countries/ from final_v1")
    parser.add_argument("--matrix", default="configs/final_v1.yaml")
    parser.add_argument("--report", default="reports/final_v1_test/results.json")
    parser.add_argument("--b", type=int, default=B)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from .config import repo_path
    from .final_eval import with_val_calibration
    from .hierarchy import build_trees, class_names_from_data
    from .paths import resolve_output_root

    spec = yaml.safe_load(repo_path(args.matrix).read_text(encoding="utf-8"))
    spec_h = yaml.safe_load(repo_path(spec["hierarchy_matrix"]).read_text(encoding="utf-8"))
    grid = spec_h["tau_grid"]
    taus = np.round(np.arange(grid["start"], grid["stop"] + 1e-9, grid["step"]), 4)
    targets = spec_h["precision_targets"]
    trees = build_trees(spec_h, class_names_from_data())
    reported = json.loads(repo_path(args.report).read_text(encoding="utf-8"))["sets"]
    out = Path(args.out) if args.out else resolve_output_root(None) / "bootstrap_v1"
    out.mkdir(parents=True, exist_ok=True)

    result = {"b": args.b, "seed": SEED, "interval": "percentile 2.5-97.5", "sets": {}}
    sets_points, sets_draws = {}, {}
    for set_name, runs in (("test", sorted(set(spec["anytime_runs"]) | set(spec["hierarchy_runs"]))),
                           ("unseen_countries", sorted(spec["unseen_runs"]))):
        per_run = {}
        for run in runs:
            saved = dict(np.load(Path(args.val_predictions) / f"{run}.npz"))
            final = dict(np.load(Path(args.final_predictions) / f"predictions_{set_name}" / f"{run}.npz"))
            data = with_val_calibration(saved, final)
            hier = tree_areas(data, trees, taus, targets) if run in spec["hierarchy_runs"] else {}
            per_run[run] = (stopping_tables(data), hier)
            print(f"{set_name}: prepared {run}", flush=True)
        point, draws = run_set(per_run, args.b, SEED)
        worst = check_against_report(point, reported[set_name])
        print(f"{set_name}: largest difference from the reported numbers {worst:.2e}", flush=True)
        if worst > CHECK:
            raise SystemExit(f"{set_name}: rebuilt numbers differ from reports/ by {worst}: stopping")
        per_run_ci = {run: {k: {"value": point[run][k], "ci95": interval(v)} for k, v in d.items()}
                      for run, d in draws.items()}
        result["sets"][set_name] = {
            "drawings": int(len(next(iter(per_run.values()))[0]["end_correct"])),
            "check_largest_difference": worst,
            "anytime_comparisons": compare_rows(point, draws, anytime_rows(spec)),
            "hierarchy_comparisons": compare_rows(point, draws, hierarchy_rows(spec, targets)),
            "per_run": per_run_ci}
        sets_points[set_name], sets_draws[set_name] = point, draws

    # Unseen countries minus test, same model (independent drawings, so independent resamples).
    shift = []
    for cond in sorted({condition(r) for r in spec["unseen_runs"]}):
        for key in ("end_accuracy", "auc/pmax", "reward/0.95/semantic"):
            pu, pt = seed_mean(sets_points["unseen_countries"], cond, key), seed_mean(sets_points["test"], cond, key)
            if pu is None or pt is None:
                continue
            diff = seed_mean(sets_draws["unseen_countries"], cond, key) - seed_mean(sets_draws["test"], cond, key)
            shift.append({"name": f"{cond} {key}: unseen minus test", "difference": float(pu - pt), "ci95": interval(diff)})
    result["unseen_minus_test"] = shift
    (out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    for set_name, s in result["sets"].items():
        print(f"\n{set_name} ({s['drawings']} drawings, B={args.b}):")
        for row in s["anytime_comparisons"] + s["hierarchy_comparisons"]:
            if row["difference"] is not None:
                print(f"  {row['name']}: {row['difference']:+.4f} [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]")
    print("\nunseen minus test:")
    for row in shift:
        print(f"  {row['name']}: {row['difference']:+.4f} [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]")
    print(f"\nBOOTSTRAP COMPLETE. Records: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
