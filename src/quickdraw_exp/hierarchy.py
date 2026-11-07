"""Step 4b: answer at the right level of detail.

    PYTHONPATH=src python -m quickdraw_exp.hierarchy --matrix configs/hierarchy_eval_v1.yaml \
        --predictions <folder with the step 4a .npz files>

Reuses the checkpoint predictions of step 4a; trains nothing and opens no test
data. See configs/hierarchy_eval_v1.yaml for the rules, all fixed in advance.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

N_CLASSES = 100
N_RANDOM = 5


# ---------------------------------------------------------------- trees

class Tree:
    """Nodes above the classes, plus the classes themselves.

    ``members``: (nodes, classes) 0/1 matrix; ``reward``: per node,
    1 - log(size) / log(number of classes); ``level``: 0 class, 1 subgroup, 2 group.
    """

    def __init__(self, names, members, levels):
        self.names = list(names)
        self.members = np.asarray(members, dtype=np.float32)
        self.level = np.asarray(levels)
        size = self.members.sum(axis=1)
        if (size == 0).any():
            raise ValueError("empty node")
        self.reward = 1.0 - np.log(size) / np.log(self.members.shape[1])

    @classmethod
    def from_groups(cls, groups: dict, class_names: list[str]) -> "Tree":
        """``groups`` is {group: {subgroup: [classes]}} or {group: [classes]}."""
        index = {name: i for i, name in enumerate(class_names)}
        n = len(class_names)
        names, rows, levels, seen = [], [], [], []
        for group, inner in groups.items():
            subgroups = inner if isinstance(inner, dict) else {}
            leaves = [c for sub in subgroups.values() for c in sub] if subgroups else list(inner)
            for sub, classes in subgroups.items():
                if len(classes) > 1:  # a one-class subgroup adds nothing over the class
                    names.append(f"{group}/{sub}")
                    rows.append(_row(classes, index, n))
                    levels.append(1)
            if len(leaves) > 1:
                names.append(group)
                rows.append(_row(leaves, index, n))
                levels.append(2)
            seen += leaves
        if sorted(seen) != sorted(class_names) or len(seen) != len(set(seen)):
            raise ValueError("the tree must hold every class exactly once")
        for name in class_names:
            names.append(name)
            rows.append(_row([name], index, n))
            levels.append(0)
        return cls(names, np.array(rows), levels)

    def shuffled(self, seed: int) -> "Tree":
        """The same shape with the classes assigned to it at random."""
        perm = np.random.default_rng(seed).permutation(self.members.shape[1])
        members = self.members.copy()
        upper = self.level > 0
        members[upper] = members[upper][:, perm]
        return Tree(self.names, members, self.level)


def _row(classes, index, n):
    row = np.zeros(n, dtype=np.float32)
    for c in classes:
        row[index[c]] = 1.0
    return row


def flat_tree(n: int) -> Tree:
    return Tree([str(i) for i in range(n)], np.eye(n), np.zeros(n, dtype=int))


# ---------------------------------------------------------------- answering and scoring

def answer(node_probs: np.ndarray, tree: Tree, tau: float) -> np.ndarray:
    """Node answered at each row, or -1 for silence: among nodes reaching tau,
    the most specific one (highest reward), ties broken by probability.
    Compared in float64: NumPy keeps float32 when comparing with a Python float,
    which would let a threshold behave differently in calibration and evaluation."""
    node_probs = np.asarray(node_probs, dtype=np.float64)
    ok = node_probs >= float(tau)
    key = np.where(ok, tree.reward[None, :] * 10.0 + node_probs, -np.inf)
    best = key.argmax(axis=1)
    return np.where(ok.any(axis=1), best, -1)


def score(nodes: np.ndarray, labels: np.ndarray, tree: Tree) -> tuple[np.ndarray, np.ndarray]:
    """Per row: whether an answer was given and is right, and its reward."""
    answered = nodes >= 0
    safe = np.where(answered, nodes, 0)
    right = answered & (tree.members[safe, labels] > 0)
    return right, np.where(right, tree.reward[safe], 0.0)


def reward_area(reward: np.ndarray, ink: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Per drawing: area under reward against ink drawn, on [0, 1]."""
    nxt = np.append(ink[1:], 1.0)
    nxt[ends - 1] = 1.0
    width = np.clip(nxt - ink, 0.0, None)
    return np.add.reduceat(reward * width, starts)


def reward_at(reward: np.ndarray, ink: np.ndarray, starts: np.ndarray, ends: np.ndarray, at: float) -> np.ndarray:
    """Per drawing: the reward in force once ``at`` of the ink is drawn (0 before the first checkpoint)."""
    out = np.zeros(len(starts))
    for i, (s, e) in enumerate(zip(starts, ends)):
        reached = np.flatnonzero(ink[s:e] <= at + 1e-12)
        if len(reached):
            out[i] = reward[s + reached[-1]]
    return out


def calibrate(probs, labels, tree: Tree, taus, target: float) -> float:
    """Lowest tau whose answers are right at least ``target`` of the time."""
    node_probs = probs @ tree.members.T
    for tau in sorted(taus):
        right, _ = score(answer(node_probs, tree, tau), labels, tree)
        answered = (answer(node_probs, tree, tau) >= 0).sum()
        if answered and right.sum() / answered >= target:
            return float(tau)
    return float(max(taus))


def evaluate(part: dict, tree: Tree, tau: float) -> dict:
    node_probs = part["probs"] @ tree.members.T
    nodes = answer(node_probs, tree, tau)
    right, reward = score(nodes, part["label"], tree)
    answered = nodes >= 0
    starts, ends = part["starts"], part["ends"]
    levels = {f"level_{k}": float((answered & (tree.level[np.where(answered, nodes, 0)] == k)).mean())
              for k in sorted(set(tree.level.tolist()))}
    return {"tau": tau,
            "reward_area": float(reward_area(reward, part["ink"], starts, ends).mean()),
            "reward_at_0.25": float(reward_at(reward, part["ink"], starts, ends, 0.25).mean()),
            "reward_at_0.5": float(reward_at(reward, part["ink"], starts, ends, 0.5).mean()),
            "precision": float(right.sum() / max(answered.sum(), 1)),
            "answered": float(answered.mean()), "answers_by_level": levels}


def halves(data: dict) -> dict:
    from .anytime import group_bounds
    out = {}
    for name, flag in (("calib", 0), ("eval", 1)):
        mask = data["half"] == flag
        part = {k: data[k][mask] for k in ("probs", "label", "ink", "drawing")}
        part["probs"] = part["probs"].astype(np.float32)
        part["starts"], part["ends"] = group_bounds(part["drawing"])
        out[name] = part
    return out


def run_trees(data: dict, trees: dict, taus, targets) -> dict:
    parts = halves(data)
    report = {}
    for target in targets:
        for name, tree_list in trees.items():
            results = []
            for tree in tree_list:
                tau = calibrate(parts["calib"]["probs"], parts["calib"]["label"], tree, taus, target)
                results.append(evaluate(parts["eval"], tree, tau))
            keys = ("reward_area", "reward_at_0.25", "reward_at_0.5", "precision", "answered")
            summary = {k: float(np.mean([r[k] for r in results])) for k in keys}
            summary["tau"] = [r["tau"] for r in results]
            summary["answers_by_level"] = results[0]["answers_by_level"]
            report.setdefault(str(target), {})[name] = summary
    return report


# ---------------------------------------------------------------- main

def class_names_from_data(root=None) -> list[str]:
    """Class names in label order, from the validation metadata."""
    from .config import repo_path
    from .data import load_holdout, open_split
    from .paths import resolve_data_root
    import yaml as _yaml
    base = _yaml.safe_load(repo_path("configs/base.yaml").read_text())
    val = open_split("val", resolve_data_root(root), load_holdout(repo_path(base["data"]["holdout_file"])))
    pairs = val.meta[["label", "word"]].drop_duplicates().sort_values("label")
    if pairs["label"].tolist() != list(range(N_CLASSES)):
        raise ValueError("labels are not 0..99")
    return pairs["word"].tolist()


def build_trees(spec: dict, class_names: list[str]) -> dict:
    import pandas as pd
    from .config import repo_path
    tree_file = yaml.safe_load(repo_path(spec["tree"]).read_text(encoding="utf-8"))
    semantic = Tree.from_groups(tree_file["groups"], class_names)
    table = pd.read_csv(repo_path("reports/eda/tables/class_summary.csv"))
    clusters = {}
    for word, cluster in zip(table["word"], table["cluster"]):
        clusters.setdefault(f"cluster {cluster}", []).append(word)
    shape = Tree.from_groups(clusters, class_names)
    return {"flat": [flat_tree(len(class_names))], "semantic": [semantic],
            "random": [semantic.shuffled(seed) for seed in range(N_RANDOM)], "shape": [shape]}


def compare(per_run: dict, comparisons: list[dict], target: str) -> list[dict]:
    rows = []
    for c in comparisons:
        a = [per_run[r][target][c["a"]]["reward_area"] for r in c["runs"] if r in per_run]
        b = [per_run[r][target][c["b"]]["reward_area"] for r in c["runs"] if r in per_run]
        row = {"name": c["name"], "precision_target": target, "runs": [r for r in c["runs"] if r in per_run]}
        if a and len(a) == len(b):
            row.update({"a_mean": float(np.mean(a)), "b_mean": float(np.mean(b)),
                        "difference": float(np.mean(a) - np.mean(b)), "a_values": a, "b_values": b})
        else:
            row["difference"] = None
        rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--predictions", required=True, help="folder holding <run>.npz from step 4a")
    parser.add_argument("--out", default=None, help="default: runs/hierarchy_v1 (or /kaggle/working/runs/...)")
    args = parser.parse_args(argv)

    from .paths import resolve_output_root
    spec = yaml.safe_load(Path(args.matrix).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else resolve_output_root(None) / spec["experiment"]
    out.mkdir(parents=True, exist_ok=True)
    trees = build_trees(spec, class_names_from_data())
    grid = spec["tau_grid"]
    taus = np.round(np.arange(grid["start"], grid["stop"] + 1e-9, grid["step"]), 4)
    targets = spec["precision_targets"]

    per_run, missing = {}, []
    for run in spec["runs"]:
        path = Path(args.predictions) / f"{run}.npz"
        if not path.is_file():
            missing.append(run)
            continue
        per_run[run] = run_trees(dict(np.load(path)), trees, taus, targets)
        print(f"scored {run}", flush=True)

    result = {"experiment": spec["experiment"], "missing_runs": missing,
              "comparisons": [row for t in targets for row in compare(per_run, spec["comparisons"], str(t))],
              "per_run": per_run}
    (out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "precision_target", "tree", "tau", "reward_area", "reward_at_0.25", "reward_at_0.5",
                         "precision", "answered"])
        for run, by_target in per_run.items():
            for target, by_tree in by_target.items():
                for tree, r in by_tree.items():
                    writer.writerow([run, target, tree, r["tau"], r["reward_area"], r["reward_at_0.25"],
                                     r["reward_at_0.5"], r["precision"], r["answered"]])
    print(json.dumps({k: result[k] for k in ("missing_runs", "comparisons")}, indent=1))
    print(f"\nHIERARCHY EVALUATION {'COMPLETE' if not missing else 'INCOMPLETE'}. Records: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
