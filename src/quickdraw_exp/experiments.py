"""Run a matrix of training jobs across the available GPUs and collect results.

    PYTHONPATH=src python -m quickdraw_exp.experiments --matrix configs/leak_v1.yaml --max-minutes 200

Jobs live in fixed folders under ``runs/<experiment>/``. Finished jobs are skipped,
so a repeated command continues where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from .paths import resolve_output_root

MIN_MINUTES_TO_START = 3.0


def build_jobs(matrix: dict) -> list[dict]:
    """Speed runs first, then every condition for each seed, seed by seed, so a
    shortened session still leaves complete comparisons for the first seed."""
    jobs = []
    for name, spec in matrix.get("speed_runs", {}).items():
        jobs.append({"name": name, "condition": name, "seed": matrix["seeds"][0], "kind": "speed", "config": spec["config"],
                     "set": [*matrix.get("speed_common", []), *spec.get("set", [])]})
    for seed in matrix["seeds"]:
        for name, spec in matrix["conditions"].items():
            jobs.append({"name": f"{name}_s{seed}", "condition": name, "seed": seed, "kind": "condition",
                         "config": spec["config"], "set": [*matrix.get("common", []), *spec.get("set", [])]})
    names = [job["name"] for job in jobs]
    if len(set(names)) != len(names):
        raise ValueError("job names repeat")
    return jobs


def read_summary(run_dir: Path) -> dict | None:
    path = run_dir / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def is_finished(run_dir: Path) -> bool:
    summary = read_summary(run_dir)
    return bool(summary and summary.get("finished"))


def job_command(job: dict, run_dir: Path, minutes_left: float, extra=()) -> list[str]:
    settings = [*job["set"], f"run.seed={job['seed']}", f"run.name={job['name']}",
                f"budget.max_minutes={minutes_left:.1f}", *extra]
    cmd = [sys.executable, "-m", "quickdraw_exp.engine", "--config", job["config"], "--run-dir", str(run_dir)]
    for item in settings:
        cmd += ["--set", item]
    return cmd


def run_jobs(jobs: list[dict], top: Path, gpus: int, max_minutes: float, extra=()) -> None:
    deadline = time.monotonic() + 60.0 * max_minutes
    pending: queue.Queue = queue.Queue()
    for job in jobs:
        if is_finished(top / job["name"]):
            print(f"skip {job['name']}: already finished", flush=True)
        else:
            pending.put(job)

    def work(gpu: int | None) -> None:
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            left = (deadline - time.monotonic()) / 60.0
            if left < MIN_MINUTES_TO_START:
                print(f"not started {job['name']}: {left:.1f} minutes left", flush=True)
                continue
            run_dir = top / job["name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            print(f"start {job['name']} on {'GPU ' + str(gpu) if gpu is not None else 'CPU'}", flush=True)
            with open(top / f"{job['name']}.log", "a", encoding="utf-8") as log:
                code = subprocess.run(job_command(job, run_dir, left, extra), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            state = "finished" if is_finished(run_dir) else f"NOT finished, exit code {code}"
            print(f"end {job['name']}: {state}, {(time.monotonic() - started) / 60:.1f} minutes", flush=True)

    threads = [threading.Thread(target=work, args=(i if gpus else None,)) for i in range(max(gpus, 1))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _row(job: dict, frac: str, scores: dict) -> dict:
    return {"job": job["name"], "condition": job["condition"], "seed": job["seed"], "kind": job["kind"], "frac": frac,
            **{k: scores[k] for k in ("n", "top1", "top3", "macro_f1", "nll", "ece")}}


def collect(jobs: list[dict], top: Path) -> list[dict]:
    rows = []
    for job in jobs:
        summary = read_summary(top / job["name"])
        if summary and summary.get("finished") and summary.get("val"):
            rows += [_row(job, frac, scores) for frac, scores in summary["val"]["by_frac"].items()]
    return rows


def summarise(rows: list[dict], early_fracs) -> dict:
    """Mean and spread over seeds per condition and fraction, plus early top-1."""
    early = [str(f) for f in early_fracs]
    out = {}
    for condition in sorted({r["condition"] for r in rows if r["kind"] == "condition"}):
        mine = [r for r in rows if r["condition"] == condition]
        seeds = sorted({r["seed"] for r in mine})
        by_frac = {}
        for frac in sorted({r["frac"] for r in mine}, key=float):
            at = [r for r in mine if r["frac"] == frac]
            by_frac[frac] = {m: {"mean": float(np.mean([r[m] for r in at])), "std": float(np.std([r[m] for r in at]))}
                             for m in ("top1", "macro_f1", "ece")}
        per_seed = [float(np.mean([r["top1"] for r in mine if r["seed"] == s and r["frac"] in early])) for s in seeds]
        out[condition] = {"seeds": seeds, "by_frac": by_frac,
                          "early_top1": {"mean": float(np.mean(per_seed)), "per_seed": per_seed}}
    return out


def decide(table: dict, margin: float, seeds_expected: int) -> dict:
    """Apply the rule written in the matrix. Needs every sequence condition on every seed."""
    needed = ("gru_prefix", "gru_resampled")
    if not all(len(table.get(c, {}).get("seeds", [])) == seeds_expected for c in needed):
        return {"sequence_input": None, "reason": "incomplete: sequence conditions are missing seeds"}
    prefix, resampled = table["gru_prefix"]["early_top1"]["mean"], table["gru_resampled"]["early_top1"]["mean"]
    gap = prefix - resampled
    choice = "resampled" if resampled >= prefix - margin else "prefix"
    result = {"sequence_input": choice, "raster_input": "prefix", "prefix_minus_resampled": gap}
    for model in ("gru", "cnn"):
        full, own = table.get(f"{model}_full_bbox"), table.get(f"{model}_prefix")
        if full and own:
            result[f"{model}_full_bbox_minus_prefix"] = full["early_top1"]["mean"] - own["early_top1"]["mean"]
    return result


def speed_report(jobs: list[dict], top: Path) -> dict:
    report = {}
    for job in jobs:
        if job["kind"] == "speed":
            summary = read_summary(top / job["name"])
            if summary and summary.get("val"):
                report[job["name"]] = {
                    "samples_per_s": summary.get("train_samples_per_s"),
                    "parameters": summary.get("parameters"),
                    "peak_gpu_memory_mb": summary.get("peak_gpu_memory_mb"),
                    "top1_by_frac": {f: round(s["top1"], 4) for f, s in summary["val"]["by_frac"].items()},
                    "mean_top1": summary["val"]["mean_top1"]}
    return report


def write_results(rows: list[dict], result: dict, top: Path) -> None:
    (top / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if rows:
        with open(top / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--max-minutes", type=float, required=True)
    parser.add_argument("--output-root", help="default: runs/, or /kaggle/working/runs on Kaggle")
    parser.add_argument("--set", action="append", default=[], metavar="section.key=value", help="added to every job")
    args = parser.parse_args(argv)

    import torch

    matrix = yaml.safe_load(Path(args.matrix).read_text(encoding="utf-8"))
    top = resolve_output_root(args.output_root) / matrix["experiment"]
    top.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(matrix)
    run_jobs(jobs, top, torch.cuda.device_count(), args.max_minutes, args.set)

    rows = collect(jobs, top)
    decision = matrix["decision"]
    table = summarise(rows, decision["early_fracs"])
    verdict = decide(table, decision["margin"], len(matrix["seeds"]))
    missing = [job["name"] for job in jobs if not is_finished(top / job["name"])]
    result = {"experiment": matrix["experiment"], "complete": not missing, "missing": missing,
              "decision": verdict, "rule": decision["rule"], "speed": speed_report(jobs, top), "table": table}
    write_results(rows, result, top)
    print(json.dumps(result, indent=2))
    print(f"\n{matrix['experiment']} {'COMPLETE' if not missing else 'INCOMPLETE'}. Records: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
