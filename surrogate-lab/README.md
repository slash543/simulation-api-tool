# SurrogateLab

A modular, open-source ML pipeline for building surrogate models of mechanical
finite element simulations. Predicts **contact pressure fields** from FEBio
`.xplt` simulation outputs using a config-driven MLP with full MLflow
experiment tracking.

Training data is produced by [xplt-parser](../xplt-parser/), which parses the
raw simulation files and exports a ready-to-use CSV. The two repositories are
designed to work together but are independently usable.

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Prepare Your Data](#prepare-your-data)
- [Quick Start](#quick-start)
- [Full Pipeline Walkthrough](#full-pipeline-walkthrough)
  - [Step 1: Configure](#step-1-configure)
  - [Step 2: Load and inspect data](#step-2-load-and-inspect-data)
  - [Step 3: Engineer features and split](#step-3-engineer-features-and-split)
  - [Step 4: Build and train](#step-4-build-and-train)
  - [Step 5: Evaluate and visualise](#step-5-evaluate-and-visualise)
  - [Step 6: View results in MLflow](#step-6-view-results-in-mlflow)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Extending the Pipeline](#extending-the-pipeline)
  - [Add a new input variable](#add-a-new-input-variable)
  - [Change the prediction target](#change-the-prediction-target)
  - [Add a custom model](#add-a-custom-model)
  - [Add a new evaluation metric](#add-a-new-evaluation-metric)
- [Discover Available Features](#discover-available-features)
- [Running Tests](#running-tests)
- [CLI Reference](#cli-reference)
- [License](#license)

---

## Overview

SurrogateLab replaces expensive FEBio finite element solves with a fast neural
network. Given per-facet geometry and the catheter insertion depth at a given
timestep, the model predicts the contact pressure at each facet — turning a
multi-minute simulation into a millisecond inference.

| Property | Detail |
|---|---|
| Inputs | `centroid_x/y/z`, `facet_area`, `insertion_depth` |
| Output | `contact_pressure` (scalar per facet) |
| Model | Configurable MLP (depth, width, activation, dropout) |
| Data source | xplt-parser `df_surrogate()` CSV |
| Tracking | MLflow — params, metrics, artifacts, model registry |
| Deployment | CLI script compatible with Azure ML |
| Licenses | Apache 2.0 / BSD / MIT only — commercial-use safe |

---

## Getting Started

### Prerequisites

- Python 3.10+
- `make` (optional but recommended)
- A CUDA-capable GPU (optional — falls back to CPU automatically)
- [xplt-parser](../xplt-parser/) to generate training CSVs from `.feb`/`.xplt`
  files

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd surrogate-lab

# Create a virtual environment and install dependencies
make env

# Or manually:
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For notebook support:

```bash
make env-full
# or: pip install jupyter ipykernel
```

### Prepare Your Data

Training data comes from [xplt-parser](../xplt-parser/). Each simulation is
exported as a single CSV using `SimulationCase.df_surrogate()` in the parser.

**Generating the CSV from xplt-parser:**

```python
# In xplt-parser/xplt_explorer.ipynb (cell 17) or your own script:
import xplt_core as xc

case = xc.SimulationCase("my_sim.feb", "my_sim.xplt", label="sim_01")
df = case.df_surrogate()
df.to_csv("sim_01_surrogate.csv", index=False)
```

`insertion_depth` is computed automatically from the prescribed displacement
boundary conditions in the `.feb` file — no manual configuration needed.
Changing the `.feb` and re-exporting will always produce correct depths.

**CSV schema — one row per (facet × timestep):**

| Column | Type | Units | Description |
|---|---|---|---|
| `centroid_x` | float | mm | Facet centroid X coordinate |
| `centroid_y` | float | mm | Facet centroid Y coordinate |
| `centroid_z` | float | mm | Facet centroid Z coordinate |
| `facet_area` | float | mm² | Facet surface area |
| `insertion_depth` | float | mm | Catheter insertion depth at this timestep |
| `contact_pressure` | float | MPa | Contact pressure (the prediction target) |

**Place CSV files in `data/simulations/`:**

```bash
cp /path/to/sim_*_surrogate.csv data/simulations/
```

The pipeline will load and concatenate all files matching
`config.yaml → data.file_pattern` (default `*.csv`) automatically.

---

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Train with default config
make train
# or: python scripts/train.py

# View results in MLflow UI
make mlflow
# open http://localhost:5000
```

---

## Full Pipeline Walkthrough

The same steps are available interactively in `notebooks/training.ipynb`.

### Step 1: Configure

Everything is controlled by `configs/config.yaml`. Verify the paths and
hyperparameters before training:

```yaml
data:
  source: "data/simulations/"
  file_pattern: "*.csv"

features:
  inputs:
    - centroid_x
    - centroid_y
    - centroid_z
    - facet_area
    - insertion_depth
  target: contact_pressure
  normalization:
    method: standard      # 'standard' (z-score) or 'minmax'

model:
  type: MLP
  layers: [128, 128, 64]
  activation: relu
  dropout: 0.0

training:
  lr: 0.001
  epochs: 200
  batch_size: 256
  early_stopping:
    enabled: true
    patience: 20

mlflow:
  experiment_name: surrogate-lab
  tracking_uri: mlruns/
```

No Python code changes are needed — the pipeline reads this file at runtime.

### Step 2: Load and inspect data

```python
from src.utils.config import load_config
from src.data.loader import load_simulation_data

cfg = load_config("configs/config.yaml")

# Loads all CSVs from data/simulations/, validates required columns,
# returns a single concatenated DataFrame.
df = load_simulation_data(cfg)

print(df.shape)          # (n_facets × n_timesteps × n_simulations, 6)
print(df.describe())
print(df["contact_pressure"].describe())
```

### Step 3: Engineer features and split

```python
from src.features.engineer import FeaturePipeline, build_xy
from src.features.splitter import split_data

# Extract feature matrix X and target vector y
X, y = build_xy(df, cfg)
print(f"X shape: {X.shape}")   # (n_rows, 5)
print(f"y shape: {y.shape}")   # (n_rows,)

# 70 / 15 / 15 train / val / test split
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, cfg)

# Fit scalers on training data only (prevents data leakage)
pipeline = FeaturePipeline(cfg)
X_train_s, y_train_s = pipeline.fit_transform(X_train, y_train)
X_val_s,   y_val_s   = pipeline.transform(X_val,   y_val)
X_test_s,  y_test_s  = pipeline.transform(X_test,  y_test)

# Save scalers for inference
pipeline.save("artifacts/scalers/")
```

### Step 4: Build and train

```python
import torch
from src.models.factory import build_model
from src.training.trainer import train

device = "cuda" if torch.cuda.is_available() else "cpu"

# Input dimension is inferred from X_train_s automatically
model = build_model(input_dim=X_train_s.shape[1], cfg=cfg)

model = train(
    model, X_train_s, y_train_s,
    X_val_s, y_val_s,
    cfg=cfg,
    run_name="my-first-run",
)
```

Training output:

```
Epoch  10/200  train_loss=0.04821  val_loss=0.04612
Epoch  20/200  train_loss=0.02934  val_loss=0.02801
...
Early stopping triggered (patience=20 exceeded).
Restored best model from checkpoints/best_model.pt (val_loss=0.00282)
```

### Step 5: Evaluate and visualise

```python
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualization import plot_predicted_vs_actual, plot_residuals

# Metrics in original physical units (inverse-scaled)
metrics = evaluate_model(model, X_test_s, y_test_s, pipeline, device)
print(f"RMSE : {metrics['rmse']:.4f} MPa")
print(f"MAE  : {metrics['mae']:.4f} MPa")
print(f"R²   : {metrics['r2']:.4f}")

# Diagnostic plots
import torch, numpy as np
model.eval()
with torch.no_grad():
    y_pred_s = model(torch.tensor(X_test_s).to(device)).cpu().numpy()

y_pred = pipeline.inverse_transform_y(y_pred_s)
y_true = pipeline.inverse_transform_y(y_test_s)

plot_predicted_vs_actual(y_true, y_pred, save_path="artifacts/pred_vs_actual.png")
plot_residuals(y_true, y_pred, save_path="artifacts/residuals.png")
```

Two diagnostic plots are produced:

- **Predicted vs Actual** — scatter along the ideal diagonal; spread = error
- **Residuals** — residuals vs predicted + histogram; ideally centred at zero

### Step 6: View results in MLflow

```bash
make mlflow
# or: mlflow ui --backend-store-uri mlruns/
```

Open [http://localhost:5000](http://localhost:5000). You will see:

- **Parameters**: model architecture, learning rate, batch size, normalization
- **Metrics**: `train_loss` and `val_loss` per epoch; final RMSE, MAE, R² on test set
- **Artifacts**: best model checkpoint, diagnostic plots, scaler files
- **Runs**: compare experiments side-by-side

To register a model after training:

```yaml
# configs/config.yaml
mlflow:
  register_model: true
  model_name: contact_pressure_surrogate
```

Promote from `Staging` to `Production` in the MLflow UI.

---

## Project Structure

```
surrogate-lab/
├── configs/
│   └── config.yaml          # Single source of truth — edit this, not code
│
├── src/
│   ├── data/
│   │   ├── loader.py        # load_simulation_data(), list_available_features()
│   │   └── schema.py        # Column validation against config
│   ├── features/
│   │   ├── engineer.py      # FeaturePipeline, build_xy()
│   │   └── splitter.py      # split_data()
│   ├── models/
│   │   ├── mlp.py           # MLP(nn.Module) — configurable depth and width
│   │   └── factory.py       # build_model(), register_model()
│   ├── training/
│   │   └── trainer.py       # train(), EarlyStopping, MLflow logging
│   ├── evaluation/
│   │   ├── metrics.py       # compute_metrics(), evaluate_model()
│   │   └── visualization.py # Prediction and residual plots
│   └── utils/
│       ├── config.py        # load_config(), get_feature_names()
│       └── logging_utils.py # get_logger()
│
├── scripts/
│   └── train.py             # CLI training entrypoint
│
├── notebooks/
│   └── training.ipynb       # Interactive walkthrough of the full pipeline
│
├── tests/                   # pytest test suite
├── data/simulations/        # Place surrogate CSVs from xplt-parser here
├── artifacts/               # Scalers, plots (auto-created)
├── checkpoints/             # Best model weights (auto-created)
├── mlruns/                  # MLflow tracking (auto-created)
├── requirements.txt
└── Makefile
```

---

## Configuration Reference

```yaml
data:
  source: "data/simulations/"   # Directory containing surrogate CSV files
  file_pattern: "*.csv"         # Glob pattern to select files
  time_to_depth:                # Fallback only — used if a CSV has a
    method: linear              # 'time_step' column but no 'insertion_depth'.
    scale: 1.0                  # CSVs from xplt-parser already contain
                                # 'insertion_depth', so this section is skipped.

features:
  # HOW TO ADD A NEW VARIABLE — see "Extending the Pipeline" below.
  inputs:                       # Column names to use as model inputs.
    - centroid_x                # Add/remove names here; no code changes needed.
    - centroid_y
    - centroid_z
    - facet_area
    - insertion_depth
  target: contact_pressure      # Column to predict. Change for other targets.
  normalization:
    method: standard            # 'standard' (z-score) or 'minmax' (0–1)

split:
  train: 0.70
  val:   0.15
  test:  0.15
  random_seed: 42

model:
  type: MLP                     # Must match a registered model type
  layers: [128, 128, 64]        # Hidden layer widths — any depth supported
  activation: relu              # relu | tanh | elu | gelu
  dropout: 0.0                  # 0.0 = disabled

training:
  optimizer: adam
  lr: 0.001
  loss: mse
  epochs: 200
  batch_size: 256
  early_stopping:
    enabled: true
    patience: 20
    min_delta: 1.0e-6
  checkpoint:
    enabled: true
    dir: checkpoints/
    save_best: true

evaluation:
  metrics: [rmse, mae, r2]
  kfold:
    enabled: false
    n_splits: 5

mlflow:
  experiment_name: surrogate-lab
  tracking_uri: mlruns/
  log_artifacts: true
  register_model: false
  model_name: contact_pressure_surrogate
```

---

## Extending the Pipeline

### Add a new input variable

Adding a variable requires changes in **both** repositories. The steps are:

**In xplt-parser (`xplt_core.py`):**

1. Add a `_col_<name>(self)` method to `SimulationCase` that returns a 1-D
   NumPy array of length `n_timesteps × n_facets`.
2. Register it in `SimulationCase.SURROGATE_COLUMNS`:
   ```python
   SURROGATE_COLUMNS = {
       ...
       'my_new_var': ('Description [units]', '_col_my_new_var'),
   }
   ```
3. Re-run `df_surrogate()` and save the CSV — the new column appears
   automatically.

**In surrogate-lab (`configs/config.yaml`):**

4. Add the column name to `features.inputs`:
   ```yaml
   features:
     inputs:
       - ...
       - my_new_var   # ← new
   ```

No other code changes are required. The schema validator, feature extractor,
normaliser, and model all adapt to the new input count automatically.

To check what columns a CSV already exposes before editing `config.yaml`:

```python
from src.data.loader import list_available_features
info = list_available_features("data/simulations/sim_01_surrogate.csv")
print("All columns     :", info["all_columns"])
print("Suggested inputs:", info["suggested_inputs"])
print("Suggested target:", info["suggested_target"])
```

### Change the prediction target

```yaml
features:
  target: von_mises_stress   # was: contact_pressure
```

Ensure the column exists in your CSV. The pipeline adapts automatically.

### Add a custom model

```python
# src/models/my_model.py
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        # define layers using cfg['model'] params

    def forward(self, x):
        # return shape (batch,) for scalar regression
        ...
```

Register it before training:

```python
from src.models.factory import register_model
from src.models.my_model import MyModel

register_model("MyModel", MyModel)
```

Switch to it in `config.yaml`:

```yaml
model:
  type: MyModel
```

### Add a new evaluation metric

Add your function to `src/evaluation/metrics.py` and call it from
`compute_metrics()`:

```python
def mean_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))))
```

---

## Discover Available Features

Use `list_available_features()` to inspect what columns a surrogate CSV
contains before editing `config.yaml`:

```python
from src.data.loader import list_available_features

info = list_available_features("data/simulations/sim_01_surrogate.csv")
# Returns:
# {
#   "all_columns":      ["centroid_x", "centroid_y", ...],
#   "suggested_inputs": ["centroid_x", "centroid_y", "centroid_z",
#                        "facet_area", "insertion_depth"],
#   "suggested_target": ["contact_pressure"],
# }
```

The function reads only the CSV header (no data loaded) so it is fast even
for large files.

---

## Running Tests

```bash
# Full test suite
make test

# Verbose output
pytest -v

# Single module
pytest tests/test_models.py -v

# With coverage report
pytest --cov src --cov-report term-missing
```

---

## CLI Reference

```bash
# Train with default config
python scripts/train.py

# Override config file
python scripts/train.py --config experiments/high_lr.yaml

# Point to a specific CSV or directory
python scripts/train.py --data data/simulations/sim_01_surrogate.csv

# Tag the MLflow run
python scripts/train.py --run-name ablation-dropout-0.1
```

### Makefile shortcuts

| Target | Description |
|---|---|
| `make env` | Create `.venv` and install core dependencies |
| `make env-dev` | + pytest and pytest-cov |
| `make env-full` | + jupyter and ipykernel |
| `make train` | Run `scripts/train.py` with default config |
| `make test` | Run the pytest suite |
| `make mlflow` | Launch MLflow UI at localhost:5000 |
| `make clean` | Remove `.venv`, checkpoints, artifacts, mlruns |

---

## License

Apache 2.0 — see [LICENSE](LICENSE). All dependencies use MIT, BSD, or
Apache 2.0 licenses. Safe for commercial use and resale.
