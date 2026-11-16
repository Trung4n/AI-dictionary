"""Write demo/assets.json: what the demo needs besides a checkpoint, taken from reports/.

    python scripts/export_demo_assets.py

* ``classes``: the 100 class names in label order (reports/errors_v1, built from the data);
* ``tree``: the frozen semantic tree (configs/hierarchy_v1.yaml);
* ``thresholds``: per run, the max-softmax stopping thresholds calibrated on validation for an
  accuracy gap of 2% and 5%, and the tree threshold for precision 0.95 and 0.90, exactly the
  values used in the final evaluation (reports/final_v2_test_v2).
The demo can then run without the dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    errors = json.loads((REPO / "reports/errors_v1/results.json").read_text())
    classes = [row["class"] for row in errors["models"]["cnn_prefix"]["stopping"]["classes"]]
    tree = yaml.safe_load((REPO / "configs/hierarchy_v1.yaml").read_text())["groups"]
    final = json.loads((REPO / "reports/final_v2_test_v2/results.json").read_text())["sets"]["test_v2"]
    thresholds = {}
    for run, rep in final["anytime"]["per_run"].items():
        thresholds[run] = {"stop_pmax": {gap: rep["pmax"]["risk_control"][gap]["tau"] for gap in ("0.02", "0.05")},
                           "tree": None}
    for run, rep in final["hierarchy"]["per_run"].items():
        thresholds.setdefault(run, {"stop_pmax": None})["tree"] = {t: rep[t]["semantic"]["tau"][0] for t in ("0.95", "0.9")}
    if len(classes) != 100 or len(set(classes)) != 100:
        raise SystemExit("expected 100 distinct class names")
    out = {"source": "reports/errors_v1, configs/hierarchy_v1.yaml, reports/final_v2_test_v2",
           "classes": classes, "tree": tree, "thresholds": thresholds}
    path = REPO / "demo" / "assets.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}: {len(classes)} classes, thresholds for {len(thresholds)} runs")


if __name__ == "__main__":
    main()
