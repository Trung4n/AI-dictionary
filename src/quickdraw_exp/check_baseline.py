"""Step 2 check: both baselines learn on a small set and share one evaluation.

    PYTHONPATH=src python -m quickdraw_exp.check_baseline

Each model runs in its own process, on its own GPU when two are available:
tensor shapes, memorising 256 drawings, then a short training run. Prints the
measured speed, which sets the quota budget. The small-run accuracy says only
that learning works; it is not a result.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = {"cnn": "configs/baseline_cnn.yaml", "gru": "configs/baseline_gru.yaml"}
SMALL = ["train.train_subset=50000", "train.epochs=5", "train.warmup_steps=20", "train.log_every=10",
         "eval.subset_per_class=20", "eval.batch_size=512", "eval.final_per_class=20", "data.num_workers=2"]
OVERFIT_SIZE, OVERFIT_STEPS = 256, 1000
OVERFIT_TARGET, SMALL_TOP1_TARGET = 0.8, 0.05  # chance is 0.01


def check_shapes(kind: str, inputs, labels) -> dict:
    import torch
    size = labels.shape[0]
    checks = {"labels_in_range": bool(labels.min() >= 0 and labels.max() < 100)}
    if kind == "cnn":
        x = inputs[0]
        checks["raster_shape"] = list(x.shape) == [size, 1, 64, 64]
        checks["raster_range"] = bool(x.min() >= 0 and x.max() <= 1 and x.max() > 0)
        checks["inputs_finite"] = bool(torch.isfinite(x).all())
    else:
        seq, lengths = inputs
        checks["sequence_shape"] = list(seq.shape) == [size, 256, 3]
        checks["lengths_in_range"] = bool(lengths.min() >= 1 and lengths.max() <= 256)
        checks["inputs_finite"] = bool(torch.isfinite(seq).all())
        beyond = [float(seq[i, int(n):].abs().sum()) for i, n in enumerate(lengths)]
        checks["padding_is_zero"] = max(beyond) == 0.0
    return checks


def memorise(cfg: dict, train_split, device) -> float:
    """Training accuracy on a few fixed drawings. A model and loss that work
    should almost learn them by heart."""
    import numpy as np
    import torch

    from .data import SubsetSplit
    from .engine import make_view
    from .models import build_model, to_inputs

    torch.manual_seed(cfg["run"]["seed"])
    pick = np.random.default_rng(cfg["run"]["seed"]).choice(len(train_split), OVERFIT_SIZE, replace=False)
    view = make_view(cfg, SubsetSplit(train_split, np.sort(pick)), None, train=False)
    batch = next(iter(torch.utils.data.DataLoader(view, batch_size=OVERFIT_SIZE)))
    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0)
    inputs, labels = to_inputs(model.input_kind, batch, device)
    model.train()
    for _ in range(OVERFIT_STEPS):
        loss = torch.nn.functional.cross_entropy(model(*inputs), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return float((model(*inputs).argmax(1) == labels).float().mean())


def worker(kind: str, output: Path) -> None:
    import torch

    from .config import load_config
    from .engine import make_view, open_data, pick_device, run_training
    from .models import build_model, to_inputs

    cfg = load_config(CONFIGS[kind], SMALL + [f"run.output_root={output}"])
    device = pick_device(cfg["run"]["device"])
    train_split, _, info = open_data(cfg)
    result = {"kind": kind, "device": str(device), "n_train_small": info["n_train"]}

    view = make_view(cfg, train_split, "random", train=True)
    batch = next(iter(torch.utils.data.DataLoader(view, batch_size=32)))
    model = build_model(cfg["model"]).to(device)
    inputs, labels = to_inputs(model.input_kind, batch, device)
    checks = check_shapes(kind, inputs, labels)
    with torch.no_grad():
        checks["logits_shape"] = list(model(*inputs).shape) == [32, 100]
    del model

    started = time.time()
    accuracy = memorise(cfg, train_split, device)
    result["memorise_accuracy"] = accuracy
    result["memorise_seconds"] = round(time.time() - started, 1)
    checks["memorises_small_set"] = accuracy >= OVERFIT_TARGET

    started = time.time()
    summary = run_training(cfg)
    result["small_run_seconds"] = round(time.time() - started, 1)
    by_frac = summary["val"]["by_frac"]
    result["small_run"] = {k: summary.get(k) for k in ("first_loss", "last_loss", "train_samples_per_s", "parameters",
                                                       "peak_gpu_memory_mb", "run_dir")}
    result["val_top1_by_frac"] = {f: round(s["top1"], 4) for f, s in by_frac.items()}
    result["metric_keys"] = sorted(by_frac["1.0"])
    checks["loss_falls"] = summary["last_loss"] < summary["first_loss"]
    checks["beats_chance"] = by_frac["1.0"]["top1"] >= SMALL_TOP1_TARGET
    checks["more_ink_helps"] = by_frac["1.0"]["top1"] > by_frac["0.1"]["top1"]
    result["checks"] = checks
    (output / f"{kind}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def budget(results: dict) -> dict:
    """Minutes per full epoch at the measured speed, for ~2.25M training drawings."""
    n_train = 2_400_000 * (1 - 0.0619)
    estimate = {}
    for kind, result in results.items():
        rate = result["small_run"]["train_samples_per_s"]
        estimate[kind] = {"samples_per_s": round(rate), "full_epoch_minutes": round(n_train / rate / 60, 1)}
    return estimate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=list(CONFIGS), help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.worker, Path(args.output))
        return 0

    import torch

    from .paths import resolve_output_root
    from .runlog import RunLogger, new_run_dir

    from .config import load_config
    cfg = load_config(CONFIGS["cnn"])
    cfg["run"]["name"] = "check_baseline"
    top = new_run_dir(resolve_output_root(cfg["run"]["output_root"]), "check_baseline", cfg["run"]["seed"])
    logger = RunLogger(top, cfg)
    gpus = torch.cuda.device_count()
    jobs = {}
    for i, kind in enumerate(CONFIGS):
        env = dict(os.environ)
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(i % gpus)
        log = open(top / f"{kind}.log", "w", encoding="utf-8")
        cmd = [sys.executable, "-m", "quickdraw_exp.check_baseline", "--worker", kind, "--output", str(top)]
        jobs[kind] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log)
        logger.event(f"started {kind} on {'GPU ' + str(i % gpus) if gpus else 'CPU'}")
        if gpus < 2:
            jobs[kind][0].wait()

    results, crashed = {}, []
    for kind, (process, log) in jobs.items():
        code = process.wait()
        log.close()
        path = top / f"{kind}.json"
        if code != 0 or not path.is_file():
            crashed.append(kind)
            print(f"\n--- {kind} crashed, last lines of its log ---")
            print("\n".join((top / f"{kind}.log").read_text(encoding="utf-8").splitlines()[-30:]))
        else:
            results[kind] = json.loads(path.read_text(encoding="utf-8"))

    same_metrics = len(results) == 2 and results["cnn"]["metric_keys"] == results["gru"]["metric_keys"]
    passed = not crashed and same_metrics and all(all(r["checks"].values()) for r in results.values())
    summary = {"passed": passed, "crashed": crashed, "same_evaluation": same_metrics, "gpus": gpus,
               "torch": torch.__version__, "budget": budget(results) if len(results) == 2 else None,
               "results": results}
    logger.finish("passed" if passed else "failed", **summary)
    print(json.dumps(summary, indent=2))
    print(f"\nSTEP 2 CHECK {'PASSED' if passed else 'FAILED'}. Records: {top}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
