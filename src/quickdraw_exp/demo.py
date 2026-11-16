"""Live guessing for the demo: strokes from a drawing canvas in, a guess out.

    from quickdraw_exp.demo import Guesser
    g = Guesser("<package>/runs/cnn_imagenet_s42")          # a run folder from the checkpoint package
    g.guess([[[10, 40, 80], [20, 22, 25]], [[50, 52], [5, 90]]])   # strokes: [[xs], [ys]], canvas pixels

The canvas strokes go through the published Quick, Draw! "simplified" recipe
(quickdraw.simplify: align, scale to 255, resample at 1 px, Ramer-Douglas-Peucker at 2.0),
then through exactly the preprocessing the model was evaluated with (prefix normalisation,
64x64 raster with stroke order, or a uniformly resampled sequence). The drawing so far is
simplified on its own, so no information about the finished drawing can leak in.

The answer follows the report: answer the top class once its max softmax probability
reaches the threshold calibrated on validation for a 5% accuracy gap; otherwise answer the
most specific node of the semantic tree whose probability reaches the tree threshold
(precision 0.95), if any. Thresholds come from demo/assets.json (scripts/export_demo_assets.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[2] / "demo" / "assets.json"


def to_points(strokes) -> tuple[np.ndarray, np.ndarray]:
    """Canvas strokes ``[[xs], [ys]]`` -> simplified integer points and pen lifts, as stored in the data."""
    from quickdraw.simplify import simplify
    pts, end = [], []
    for xs, ys in strokes:
        if len(xs) == 0 or len(xs) != len(ys):
            continue
        p = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
        e = np.zeros(len(p), dtype=bool)
        e[-1] = True
        pts.append(p)
        end.append(e)
    if not pts:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    out, out_end = simplify(np.vstack(pts), np.concatenate(end))
    return np.clip(out, 0, 255).astype(np.uint8), out_end


class Guesser:
    def __init__(self, run_dir, assets=ASSETS, device: str = "cpu", gap: str = "0.05", precision: str = "0.95"):
        import torch

        from .anytime import load_run
        from .hierarchy import Tree
        from .models import is_raster
        self.torch = torch
        self.run = Path(run_dir).name
        self.device = torch.device(device)
        self.cfg, self.model = load_run(Path(run_dir), self.device)
        self.raster = is_raster(self.cfg["model"])
        a = json.loads(Path(assets).read_text(encoding="utf-8"))
        self.classes = a["classes"]
        th = a["thresholds"].get(self.run)
        if th is None or th.get("stop_pmax") is None:
            raise ValueError(f"no calibrated stopping threshold for {self.run} in {assets}")
        self.stop_tau = float(th["stop_pmax"][gap])
        self.tree = Tree.from_groups(a["tree"], self.classes)
        self.tree_tau = float(th["tree"][precision]) if th.get("tree") else None
        self.delta_std = None
        if not self.raster:
            from quickdraw.dataset import load_resampled_stats
            step = self.cfg["data"].get("resample_step")
            if step is None:
                raise ValueError("sequence runs without data.resample_step need the dataset's seq_stats.json")
            self.delta_std = load_resampled_stats(step)

    def inputs(self, pts: np.ndarray, end: np.ndarray):
        from quickdraw.transforms import normalize, rasterize, to_sequence

        from .models import to_inputs
        torch, data = self.torch, self.cfg["data"]
        if self.raster:
            img = rasterize(normalize(pts), end, data["image_size"], order_encoding=data["order_encoding"])
            batch = (torch.from_numpy(np.asarray(img))[None], torch.zeros(1, dtype=torch.long))
        else:
            seq, n = to_sequence(normalize(pts), end, data["max_len"], self.delta_std, data.get("resample_step"))
            batch = (torch.as_tensor(np.asarray(seq))[None], torch.tensor([int(n)]), torch.zeros(1, dtype=torch.long))
        return to_inputs(self.model.input_kind, batch, self.device)[0]

    def probabilities(self, strokes) -> np.ndarray | None:
        pts, end = to_points(strokes)
        if len(pts) == 0:
            return None
        with self.torch.no_grad():
            logits, _ = self.model.forward_all(*self.inputs(pts, end))
        return self.torch.softmax(logits.float(), dim=1)[0].cpu().numpy().astype(np.float64)

    def guess(self, strokes, top: int = 5) -> dict:
        from .hierarchy import answer
        probs = self.probabilities(strokes)
        if probs is None:
            return {"ready": False}
        order = np.argsort(probs)[::-1][:top]
        best = int(order[0])
        out = {"ready": True, "run": self.run, "strokes": len(strokes),
               "top": [{"class": self.classes[i], "probability": float(probs[i])} for i in order],
               "should_answer": bool(probs[best] >= self.stop_tau), "answer": None,
               "stop_threshold": self.stop_tau, "group_answer": None}
        if out["should_answer"]:
            out["answer"] = self.classes[best]
        elif self.tree_tau is not None:
            node = int(answer((probs @ self.tree.members.T)[None], self.tree, self.tree_tau)[0])
            if node >= 0:
                level = int(self.tree.level[node])
                name = self.tree.names[node] if level > 0 else self.classes[int(np.argmax(self.tree.members[node]))]
                out["group_answer"] = {"name": name, "level": ["class", "subgroup", "group"][level],
                                       "probability": float(probs @ self.tree.members[node])}
        return out
