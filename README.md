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

## Layout

```
src/quickdraw/   io.py, stats.py (EDA statistics), prefix.py, viz.py (EDA helpers),
                 transforms.py, prepare.py, dataset.py (preprocessing pipeline),
                 simplify.py, leak.py (leak audit)
scripts/         fit_resampled_stats.py, leak_audit.py, raw_pilot.py
notebooks/       00_dataset_collection.ipynb, 01_eda.ipynb, 02_preprocessing.ipynb
tests/           unit tests for transforms, checks on the processed data
reports/         generated figures and tables
```

## Attribution

Dataset © Google, released under CC BY 4.0 via the Quick, Draw! project. All code in
`src/`, `scripts/`, `notebooks/` and `tests/` is our own; it uses NumPy, pandas, PyArrow, OpenCV,
Matplotlib, scikit-learn and PyTorch.
