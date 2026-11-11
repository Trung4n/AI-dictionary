# AI-dictionary — QuickDraw doodle recognition (CO3133 Deep Learning project)

Data: a 100-class × 30,000-drawing subset of the Kaggle competition
[Quick, Draw! Doodle Recognition](https://www.kaggle.com/competitions/quickdraw-doodle-recognition)
(`train_simplified`). The class choice and its rationale are in `dataset/class_selection/`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
```

Python 3.12. The data files are not in git: place `quickdraw_100c_30k.parquet`,
`quickdraw_100c_30k_classes.txt` and `class_selection/` under `dataset/`
(see `notebooks/00_dataset_collection.ipynb` for how they were produced from Kaggle).

## Run order

| Step | Command / notebook | Output |
|---|---|---|
| 1. Data analysis | `notebooks/01_eda.ipynb` | `reports/eda/` (figures, tables, decisions in section 13); cache `dataset/derived/drawing_stats.parquet` |
| 2. Clean, split, pack | `PYTHONPATH=src .venv/Scripts/python -m quickdraw.prepare` (~ a few minutes, 4 processes) | `dataset/processed/`, `reports/preprocessing/manifest.json` |
| 3. Check preprocessing | `.venv/Scripts/python -m pytest` and `notebooks/02_preprocessing.ipynb` | `reports/preprocessing/` |
| 4. (Optional) Leak audit | `.venv/Scripts/python scripts/leak_audit.py`; for the raw pilot, download `{class}.ndjson` from `https://storage.googleapis.com/quickdraw_dataset/full/raw/` into a folder, then `.venv/Scripts/python -I scripts/raw_pilot.py <folder> triangle bat` | `reports/preprocessing/leak_audit.md` explains the results |

On Windows PowerShell set the path with `$env:PYTHONPATH="src"` instead of the `PYTHONPATH=src` prefix.
`manifest.json` records the seed, a digest of the split and the sha256 of every
processed file; a rebuild must reproduce the same `split_digest`.

## Using the processed data

```python
from quickdraw.dataset import QuickDrawSplit, RasterView, SequenceView
from quickdraw.transforms import AugmentConfig, EVAL_FRACS

train = RasterView(QuickDrawSplit("train"), frac="random", augment=AugmentConfig(), size=64)
val = {f: RasterView(QuickDrawSplit("val"), frac=f) for f in EVAL_FRACS}
img, label = train[0]          # uint8 (64, 64), int
train.set_epoch(epoch)         # call once per epoch for fresh prefixes/augmentation
```

`SequenceView` returns `(seq float32 (256, 3), length, label)` in stroke-3 format.
Both are map-style datasets usable with `torch.utils.data.DataLoader`.

**Prefixes leak part of their size relative to the finished drawing**, see
`reports/preprocessing/leak_audit.md`. For early-recognition results on sequences use
`SequenceView(..., resample_step=RESAMPLE_STEP)`, state the remaining leak as a limitation, and
always report results separately by `recognized`: the 100% point means "the round stopped", not
"the drawing was finished".

## Experiments

Run from the repository root on any machine:

```bash
PYTHONPATH=src python -m quickdraw_exp.smoke --config configs/smoke.yaml   # step 1 check
PYTHONPATH=src python -m quickdraw_exp.check_baseline                      # step 2 check
PYTHONPATH=src python -m quickdraw_exp.engine --config configs/baseline_cnn.yaml   # train a model
PYTHONPATH=src python -m quickdraw_exp.engine --config configs/baseline_cnn.yaml --resume <run folder>
PYTHONPATH=src python -m quickdraw_exp.experiments --matrix configs/leak_v1.yaml --max-minutes 200   # step 3
PYTHONPATH=src python -m quickdraw_exp.experiments --matrix configs/anytime_v1.yaml --max-minutes 120  # step 4a
PYTHONPATH=src python -m quickdraw_exp.anytime --matrix configs/anytime_v1.yaml --runs runs/anytime_v1 --baselines <leak_v1 folder>
PYTHONPATH=src python -m quickdraw_exp.hierarchy --matrix configs/hierarchy_eval_v1.yaml --predictions <step 4a .npz folder>  # step 4b, CPU
PYTHONPATH=src python -m quickdraw_exp.experiments --matrix configs/followup_v1.yaml --max-minutes 300   # leak_v2, extra seeds
PYTHONPATH=src python -m quickdraw_exp.experiments --matrix configs/followup_v2.yaml --max-minutes 420   # step 4a on the uniform input
PYTHONPATH=src python -m quickdraw_exp.analysis --predictions <folder of all .npz>   # by recognized, hierarchy by group, CPU
PYTHONPATH=src python -m quickdraw_exp.final_eval --split val --val-predictions <folder>   # check, then --split test (once)
PYTHONPATH=src python -m quickdraw_exp.experiments --matrix configs/pretrained_v1.yaml --max-minutes 480   # ResNet-18 base model (validation only)
PYTHONPATH=src python -m pytest tests -k 'not processed_data'
```

Set `budget.max_minutes` to stop and save before a Kaggle session or quota ends.
Models are compared on validation data; only `final_eval.py` opens the test split, once, with `configs/final_v1.yaml`.
`configs/pretrained_v1.yaml` (torchvision ResNet-18, ImageNet weights and from scratch) was added after
that evaluation, so its models have validation results only; run it with `notebooks/kaggle_pretrained_v1.ipynb`.

The processed data is found automatically: `data.root` in the config, then the
`QUICKDRAW_DATA` environment variable, `dataset/processed/`, `dataset/`, and on
Kaggle anything attached under `/kaggle/input`. On Kaggle, use the preinstalled
packages instead of `requirements.txt`, whose pins can conflict with them; open
the `notebooks/kaggle_step*.ipynb` files.

Each run writes a folder under `runs/` (`/kaggle/working/runs` on Kaggle) with
its resolved configuration, environment, metrics, events and checkpoints, and
adds a line to `runs/index.csv`. Override settings with `--set section.key=value`;
unknown keys are an error.

`configs/country_holdout_v1.json` freezes five countries left out of training and
validation for the unseen-country test. Split v1 is unchanged: they are filtered
at load time. Test data opens only with `final_evaluation=True`.

The step 3 and step 4 runs use `input_norm: prefix` without `resample_step`
(they predate the leak audit), so their absolute early-prefix accuracies keep the
residual leak described above; comparisons between methods use the same input.
`configs/followup_v1.yaml` retrains the GRU baseline with `data.resample_step: 8.0`
(`gru_uniform`) to measure what that residual leak adds.

## Layout

```
src/quickdraw/   io.py, stats.py (EDA statistics), prefix.py, viz.py (EDA helpers),
                 transforms.py, prepare.py, dataset.py (preprocessing pipeline),
                 simplify.py, leak.py (leak audit)
src/quickdraw_exp/  config.py, paths.py, data.py, runlog.py, checkpoint.py, metrics.py, models.py, views.py,
                 engine.py (train, evaluate), experiments.py, anytime.py (when to guess), hierarchy.py, analysis.py, final_eval.py,
                 smoke.py, check_baseline.py
configs/         base.yaml, baseline*.yaml, leak_v1.yaml, anytime_v1.yaml, hierarchy_v1.yaml (frozen), hierarchy_eval*.yaml, followup_v*.yaml, final_v1.yaml,
                 smoke.yaml, country_holdout_v1.json
scripts/         fit_resampled_stats.py, leak_audit.py, raw_pilot.py
notebooks/       00_dataset_collection.ipynb, 01_eda.ipynb, 02_preprocessing.ipynb, kaggle_*.ipynb
tests/           unit tests for transforms, checks on the processed data, experiment tests
reports/         generated figures and tables
```

## Attribution

Dataset © Google, released under CC BY 4.0 via the Quick, Draw! project. `src/quickdraw/`,
`scripts/`, the analysis notebooks and the preprocessing tests are our own; they use NumPy, pandas,
PyArrow, OpenCV, Matplotlib, scikit-learn and PyTorch. `src/quickdraw_exp/`, `configs/`,
`tests/test_exp_setup.py`, `tests/test_baseline.py`, `tests/test_step3.py`, `tests/test_step4.py`,
`tests/test_step4b.py`, `tests/test_followup.py`, `tests/test_analysis.py`, `tests/test_final_eval.py`, `tests/test_pretrained.py`,
`paper/make_assets.py` and the `kaggle_*.ipynb` notebooks were written with AI assistance (Claude)
and use PyTorch, torchvision and PyYAML. The base model `TorchvisionResNet` is torchvision's ResNet-18, used
unchanged apart from its last layer, with its ImageNet weights (BSD-3-Clause code; weights from the
torchvision model zoo). `RasterResNet` follows the ResNet-18 layout of He et al. (2016); the
confidence head follows ConfidNet (Corbière et al., 2019); threshold calibration follows Learn then
Test (Angelopoulos et al., 2021) with the accuracy-gap risk of Ringel et al. (2024); the hierarchical
reward follows Deng et al. (2012).
