"""Anytime evaluation: how early can a model guess, and how well does it know when?

    PYTHONPATH=src python -m quickdraw_exp.anytime --matrix configs/anytime_v1.yaml \
        --runs runs/anytime_v1 --baselines <folder holding the leak_v1 runs>

Each drawing is checked at causal checkpoints: the end of every stroke and every
8 points, known while drawing without knowing how big the drawing will be. At each
checkpoint the prefix is normalised by its own bounding box, exactly as in
training. A stopping policy answers at the first checkpoint whose score reaches a
threshold, or at the end of the drawing. Earliness is the share of the drawing's
ink drawn by then, measured afterwards.

Validation drawings (held-out countries removed) are split once, per class, into a
calibration half that picks thresholds and an evaluation half that reports them.
The test split is not opened here.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

CHECK_EVERY = 8
MAX_CHECKS = 40
PER_CLASS = 100          # drawings per class in each half
SPLIT_SEED = 2026        # fixed: every run is scored on the same drawings
GRID = np.linspace(0.0, 1.0, 101)
GAPS = (0.02, 0.05)      # allowed accuracy lost by answering early
DELTA = 0.05


# ---------------------------------------------------------------- checkpoints

def checkpoints(end: np.ndarray, every: int = CHECK_EVERY, max_checks: int = MAX_CHECKS) -> np.ndarray:
    """Numbers of points kept at each checkpoint, ascending, the full drawing last."""
    n = len(end)
    ks = set((np.flatnonzero(end) + 1).tolist()) | set(range(every, n + 1, every)) | {n}
    ks = np.array(sorted(ks), dtype=np.int64)
    if len(ks) > max_checks:
        keep = np.unique(np.rint(np.linspace(0, len(ks) - 1, max_checks)).astype(np.int64))
        ks = ks[keep]
    return ks


def ink_fraction(pts: np.ndarray, end: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Share of the drawing's ink within the first k points; points when there is no ink."""
    d = np.hypot(*np.diff(pts.astype(np.float64), axis=0).T) if len(pts) > 1 else np.zeros(0)
    d[end[:-1]] = 0.0
    cum = np.concatenate([[0.0], np.cumsum(d)])
    if cum[-1] <= 0:
        return ks / len(pts)
    return cum[ks - 1] / cum[-1]


def build_items(split, positions) -> dict:
    """One row per (drawing, checkpoint), grouped by drawing in checkpoint order."""
    rows = {"drawing": [], "k": [], "ink": [], "strokes": [], "label": []}
    for pos in positions:
        pts, end, label = split[int(pos)]
        ks = checkpoints(end)
        done = np.cumsum(end)[ks - 1]
        rows["drawing"] += [int(pos)] * len(ks)
        rows["k"] += ks.tolist()
        rows["ink"] += ink_fraction(pts, end, ks).tolist()
        rows["strokes"] += done.tolist()
        rows["label"] += [label] * len(ks)
    return {key: np.asarray(value) for key, value in rows.items()}


def group_bounds(drawing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) of each drawing's rows; rows must be grouped."""
    change = np.flatnonzero(np.diff(drawing)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(drawing)]])
    return starts, ends


def calibration_split(labels: np.ndarray, per_class: int = PER_CLASS, seed: int = SPLIT_SEED):
    """Positions of the calibration and evaluation halves, per class."""
    rng = np.random.default_rng(seed)
    calib, evals = [], []
    for c in np.unique(labels):
        where = rng.permutation(np.flatnonzero(labels == c))
        calib.append(where[:per_class])
        evals.append(where[per_class:2 * per_class])
    return np.sort(np.concatenate(calib)), np.sort(np.concatenate(evals))


# ---------------------------------------------------------------- policies and metrics

def halt_rows(mask: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Row of the first True in each drawing, or its last row when there is none."""
    big = np.iinfo(np.int64).max
    candidates = np.where(mask, np.arange(len(mask)), big)
    first = np.minimum.reduceat(candidates, starts)
    return np.where(first == big, ends - 1, first)


def sweep(score, correct, ink, starts, ends, taus) -> dict:
    """Accuracy, mean earliness and Pictionary score for each threshold."""
    acc, early, game = [], [], []
    for tau in taus:
        rows = halt_rows(score >= tau, starts, ends)
        acc.append(correct[rows].mean())
        early.append(ink[rows].mean())
        game.append((correct[rows] * (1.0 - ink[rows])).mean())
    return {"tau": np.asarray(taus, dtype=float), "accuracy": np.array(acc), "earliness": np.array(early),
            "pictionary": np.array(game)}


def frontier(earliness: np.ndarray, accuracy: np.ndarray, grid=GRID) -> np.ndarray:
    """Best accuracy reachable with mean earliness at most each grid value (0 if none)."""
    out = np.zeros(len(grid))
    for i, g in enumerate(grid):
        ok = earliness <= g + 1e-12
        out[i] = accuracy[ok].max() if ok.any() else 0.0
    return out


def area(earliness, accuracy, grid=GRID) -> float:
    """Area under the accuracy-earliness frontier, on [0, 1]. Higher is better: high
    accuracy reached with little ink. It ignores how a threshold would be chosen."""
    return float(frontier(earliness, accuracy, grid).mean())


def upper_bound(failures: int, n: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper bound on a proportion, level 1 - delta."""
    from scipy.stats import beta
    return 1.0 if failures == n else float(beta.ppf(1 - delta, failures + 1, n - failures))


def accuracy_gap(correct, rows, ends) -> np.ndarray:
    """Per drawing: 1 when the answer at the end would be right but the early answer is wrong."""
    return correct[ends - 1] * (1.0 - correct[rows])


def calibrate(score, correct, ink, starts, ends, taus, gap: float, delta: float = DELTA) -> float:
    """Lowest threshold whose accuracy gap, the share of drawings an early answer
    gets wrong that the end of the drawing gets right, stays below ``gap`` with
    probability 1 - delta. Thresholds are tested from high to low, stopping at the
    first failure (fixed-sequence testing, as in Learn then Test; the gap is the
    risk of Ringel et al., 2024, marginal version). +inf means answer at the end."""
    chosen = np.inf
    for tau in sorted(taus, reverse=True):
        rows = halt_rows(score >= tau, starts, ends)
        failures = int(accuracy_gap(correct, rows, ends).sum())
        if upper_bound(failures, len(rows), delta) <= gap:
            chosen = tau
        else:
            break
    return float(chosen)


def thresholds(score: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(score[np.isfinite(score)], np.linspace(0, 1, 201))
    return np.unique(np.concatenate([[-np.inf], quantiles, [np.inf]]))


def policy_report(score_c, score_e, cal, ev) -> dict:
    """One scoring rule: thresholds come from calibration, numbers from evaluation."""
    taus = thresholds(score_c)
    curve = sweep(score_e, ev["correct"], ev["ink"], ev["starts"], ev["ends"], taus)
    cal_curve = sweep(score_c, cal["correct"], cal["ink"], cal["starts"], cal["ends"], taus)
    best = int(np.argmax(cal_curve["pictionary"]))
    report = {"auc": area(curve["earliness"], curve["accuracy"]),
              "accuracy_at_earliness": {str(e): float(frontier(curve["earliness"], curve["accuracy"], np.array([e]))[0])
                                        for e in (0.3, 0.5)},
              "pictionary": {"tau": float(taus[best]), "score": float(curve["pictionary"][best]),
                             "accuracy": float(curve["accuracy"][best]), "earliness": float(curve["earliness"][best])},
              "risk_control": {},
              "curve": {"earliness": curve["earliness"].round(5).tolist(), "accuracy": curve["accuracy"].round(5).tolist()}}
    for gap in GAPS:
        tau = calibrate(score_c, cal["correct"], cal["ink"], cal["starts"], cal["ends"], taus, gap)
        rows = halt_rows(score_e >= tau, ev["starts"], ev["ends"])
        realised = float(accuracy_gap(ev["correct"], rows, ev["ends"]).mean())
        report["risk_control"][str(gap)] = {"tau": tau, "gap": realised, "met": realised <= gap,
                                            "accuracy": float(ev["correct"][rows].mean()),
                                            "earliness": float(ev["ink"][rows].mean())}
    return report


def scores_from(probs: np.ndarray, confidence: np.ndarray) -> dict:
    """Scores in float64. With float32 scores, NumPy compares a Python-float
    threshold in float32 but a NumPy-float threshold in float64, so a threshold
    chosen in calibration could select different rows in evaluation."""
    p = np.sort(probs.astype(np.float64), axis=1)
    scores = {"pmax": p[:, -1], "margin": p[:, -1] - p[:, -2]}
    if np.isfinite(confidence).all():
        scores["conf"] = confidence.astype(np.float64)
    return scores


def evaluate_policies(data: dict) -> dict:
    """All policies for one run, from its saved checkpoint predictions."""
    halves = {}
    for name in ("calib", "eval"):
        mask = data["half"] == (0 if name == "calib" else 1)
        part = {key: data[key][mask] for key in ("probs", "confidence", "label", "ink", "strokes", "drawing")}
        part["correct"] = (part["probs"].argmax(axis=1) == part["label"]).astype(np.float64)
        part["starts"], part["ends"] = group_bounds(part["drawing"])
        part["scores"] = scores_from(part["probs"], part["confidence"])
        halves[name] = part
    cal, ev = halves["calib"], halves["eval"]
    out = {name: policy_report(cal["scores"][name], ev["scores"][name], cal, ev) for name in ev["scores"]}
    strokes_curve = sweep(ev["strokes"].astype(float), ev["correct"], ev["ink"], ev["starts"], ev["ends"],
                          [*range(1, 31), np.inf])
    out["fixed_strokes"] = {"auc": area(strokes_curve["earliness"], strokes_curve["accuracy"]),
                            "curve": {"earliness": strokes_curve["earliness"].round(5).tolist(),
                                      "accuracy": strokes_curve["accuracy"].round(5).tolist()}}
    oracle = halt_rows(ev["correct"] > 0, ev["starts"], ev["ends"])
    last = ev["ends"] - 1
    out["reference"] = {"oracle_first_correct": {"accuracy": float(ev["correct"][oracle].mean()),
                                                 "earliness": float(ev["ink"][oracle].mean())},
                        "end_of_drawing": {"accuracy": float(ev["correct"][last].mean())},
                        "first_checkpoint": {"accuracy": float(ev["correct"][ev["starts"]].mean()),
                                             "earliness": float(ev["ink"][ev["starts"]].mean())},
                        "drawings": int(len(ev["starts"])), "checkpoints": int(len(ev["drawing"]))}
    return out


# ---------------------------------------------------------------- running models

class CheckpointDataset:
    """The prefix at each (drawing, checkpoint) row, prepared as in training."""

    def __init__(self, split, items: dict, cfg: dict, delta_std):
        self.split, self.items, self.cfg, self.delta_std = split, items, cfg, delta_std

    def __len__(self) -> int:
        return len(self.items["k"])

    def __getitem__(self, j: int):
        from quickdraw.transforms import normalize, rasterize, to_sequence
        pts, end, label = self.split[int(self.items["drawing"][j])]
        k = int(self.items["k"][j])
        part, part_end = pts[:k], end[:k].copy()
        part_end[-1] = True
        data = self.cfg["data"]
        if self.cfg["model"]["kind"] == "cnn":
            return rasterize(normalize(part), part_end, data["image_size"], order_encoding=data["order_encoding"]), label
        seq, n = to_sequence(normalize(part), part_end, data["max_len"], self.delta_std, data.get("resample_step"))
        return seq, n, label


def sequence_scale(cfg: dict, root):
    """The delta scale the run was trained with: fitted for resampled sequences
    when ``data.resample_step`` is set, otherwise the processed data's."""
    from quickdraw.dataset import load_resampled_stats, load_seq_stats
    step = cfg["data"].get("resample_step")
    return load_resampled_stats(step) if step is not None else load_seq_stats(root)


def find_runs(folders) -> list[Path]:
    """Finished runs with a best checkpoint, at most two levels below each folder."""
    found = []
    for folder in folders:
        folder = Path(folder)
        for candidate in [folder, *folder.glob("*"), *folder.glob("*/*")]:
            summary = candidate / "summary.json"
            if (candidate / "checkpoints" / "best.pt").is_file() and summary.is_file():
                if json.loads(summary.read_text()).get("finished"):
                    found.append(candidate)
    return sorted(set(found))


def condition_of(run_dir: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)_s(\d+)", run_dir.name)
    return (match.group(1), int(match.group(2))) if match else (run_dir.name, -1)


def predict_run(run_dir: Path, device, num_workers: int = 2, batch_size: int = 1024, per_class: int = PER_CLASS) -> dict:
    import torch

    from .config import repo_path
    from .data import load_holdout, open_split
    from .models import build_model, load_weights
    from .paths import resolve_data_root

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if cfg["data"].get("input_norm", "prefix") != "prefix":
        raise ValueError(f"{run_dir.name}: anytime evaluation uses prefix normalisation only")
    model = build_model(cfg["model"]).to(device).eval()
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    load_weights(model, state["model"])

    root = resolve_data_root(None)
    val = open_split("val", root, load_holdout(repo_path(cfg["data"]["holdout_file"])))
    calib, evals = calibration_split(np.asarray(val.labels), per_class)
    parts = [build_items(val, calib), build_items(val, evals)]
    items = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    items["half"] = np.concatenate([np.zeros(len(parts[0]["k"]), np.int8), np.ones(len(parts[1]["k"]), np.int8)])

    dataset = CheckpointDataset(val, items, cfg, sequence_scale(cfg, root))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                         pin_memory=device.type == "cuda")
    from .models import to_inputs
    probs, confidence = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, _ = to_inputs(model.input_kind, batch, device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits, conf = model.forward_all(*inputs)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy().astype(np.float16))
            confidence.append(conf.float().cpu().numpy() if conf is not None else np.full(len(logits), np.nan, np.float32))
    items["probs"] = np.concatenate(probs)
    items["confidence"] = np.concatenate(confidence)
    return items


def aggregate(per_run: dict) -> dict:
    """Mean and spread over seeds of each condition's headline numbers."""
    table = {}
    for (condition, seed), report in sorted(per_run.items()):
        for policy, r in report.items():
            if policy == "reference":
                continue
            entry = table.setdefault(condition, {}).setdefault(policy, {"seeds": [], "auc": []})
            entry["seeds"].append(seed)
            entry["auc"].append(r["auc"])
            if "pictionary" in r:
                entry.setdefault("pictionary", []).append(r["pictionary"]["score"])
                for e, a in r["accuracy_at_earliness"].items():
                    entry.setdefault(f"accuracy_at_{e}", []).append(a)
                for t, rc in r["risk_control"].items():
                    entry.setdefault(f"earliness_at_gap_{t}", []).append(rc["earliness"])
                    entry.setdefault(f"met_gap_{t}", []).append(rc["met"])
    for condition in table.values():
        for entry in condition.values():
            for key in list(entry):
                if key not in ("seeds",) and not key.startswith("met_"):
                    entry[key] = {"mean": float(np.mean(entry[key])), "std": float(np.std(entry[key])), "values": entry[key]}
    return table


def compare(table: dict, comparisons: list[dict], metric: str = "auc") -> list[dict]:
    """Differences fixed in the matrix before any result was seen."""
    out = []
    for c in comparisons:
        (cond_a, pol_a), (cond_b, pol_b) = c["a"], c["b"]
        a, b = table.get(cond_a, {}).get(pol_a), table.get(cond_b, {}).get(pol_b)
        row = {"name": c["name"], "a": c["a"], "b": c["b"], "metric": metric}
        if a and b:
            row.update({"a_mean": a[metric]["mean"], "b_mean": b[metric]["mean"],
                        "difference": a[metric]["mean"] - b[metric]["mean"],
                        "a_values": a[metric]["values"], "b_values": b[metric]["values"]})
        else:
            row["difference"] = None
        out.append(row)
    return out


def main(argv=None) -> int:
    import torch

    from .engine import pick_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--runs", nargs="*", default=[], help="folders holding finished runs")
    parser.add_argument("--predictions", help="folder with saved <run>.npz: rescore them without any model")
    parser.add_argument("--baselines", nargs="*", default=[], help="folders holding the baseline runs")
    parser.add_argument("--out", help="default: <first --runs folder>/anytime")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--per-class", type=int, default=PER_CLASS, help="drawings per class in each half")
    args = parser.parse_args(argv)

    matrix = yaml.safe_load(Path(args.matrix).read_text(encoding="utf-8"))
    spec = matrix["anytime"]
    if not args.runs and not args.predictions:
        parser.error("give --runs or --predictions")
    out = Path(args.out) if args.out else Path(args.runs[0] if args.runs else "runs") / "anytime"
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(matrix["conditions"]) | set(spec["baseline_conditions"])
    if args.predictions:
        runs = sorted(p.with_suffix("") for p in Path(args.predictions).glob("*.npz"))
    else:
        runs = find_runs([*args.runs, *args.baselines])
    runs = [r for r in runs if condition_of(r)[0] in wanted]
    device = pick_device("auto")
    per_run = {}
    for run_dir in runs:
        key = condition_of(run_dir)
        saved = out / f"{run_dir.name}.npz"
        if args.predictions:
            saved = Path(args.predictions) / f"{run_dir.name}.npz"
        elif not saved.is_file():
            print(f"predicting {run_dir.name}", flush=True)
            np.savez_compressed(saved, **predict_run(run_dir, device, args.num_workers, per_class=args.per_class))
        data = dict(np.load(saved))
        per_run[key] = evaluate_policies(data)
        (out / f"{run_dir.name}.json").write_text(json.dumps(per_run[key], indent=1), encoding="utf-8")
        print(f"scored {run_dir.name}", flush=True)

    found = {condition_of(r)[0] for r in runs}
    table = aggregate(per_run)
    result = {"missing_conditions": sorted(wanted - found), "runs": [r.name for r in runs],
              "comparisons": compare(table, spec["comparisons"]),
              "table": {c: {p: {k: v for k, v in e.items()} for p, e in pols.items()} for c, pols in table.items()},
              "reference": {f"{c}_s{s}": r["reference"] for (c, s), r in per_run.items()},
              "torch": torch.__version__}
    (out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "seed", "policy", "auc", "pictionary", "accuracy_at_0.3", "accuracy_at_0.5",
                         "earliness_at_gap_0.02", "gap_0.02", "earliness_at_gap_0.05", "gap_0.05"])
        for (condition, seed), report in sorted(per_run.items()):
            for policy, r in report.items():
                if policy in ("reference", "fixed_strokes"):
                    continue
                rc = r["risk_control"]
                writer.writerow([condition, seed, policy, r["auc"], r["pictionary"]["score"],
                                 r["accuracy_at_earliness"]["0.3"], r["accuracy_at_earliness"]["0.5"],
                                 rc["0.02"]["earliness"], rc["0.02"]["gap"], rc["0.05"]["earliness"], rc["0.05"]["gap"]])
    print(json.dumps({k: result[k] for k in ("missing_conditions", "runs", "comparisons")}, indent=1))
    print(f"\nANYTIME EVALUATION {'COMPLETE' if not result['missing_conditions'] else 'INCOMPLETE'}. Records: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
