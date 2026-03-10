"""
Azure ML Pipeline (Dormant)
============================
Defines a 3-step Azure ML DSL pipeline for training the catheter insertion
pressure surrogate model at scale.

Status
------
This module is **dormant by default**.  All public functions return ``None``
and print a warning unless the environment variable ``USE_AZURE_ML=true`` is
set.

This allows the module to be imported freely in all environments (including
local dev without the Azure SDK installed) without raising errors.

Required Azure resources (when enabled)
-----------------------------------------
* **Azure ML Workspace** — with a ``workspace_config.json`` file (or set via
  environment variables ``AZURE_SUBSCRIPTION_ID``, ``AZURE_RESOURCE_GROUP``,
  ``AZURE_WORKSPACE_NAME``).
* **Compute cluster** — a CPU or GPU cluster in the workspace
  (e.g. ``"cpu-cluster"``).  The name is passed as ``compute_name``.
* **Environment** — a registered Azure ML environment containing the project
  dependencies (``requirements.txt``).  You can create one with:

      az ml environment create --name dtui-env --conda-file conda.yml

* **Datastore** — the default datastore (blob-backed) or a custom datastore
  path containing the raw simulation outputs.

Pipeline steps
--------------
1. **extract** — run ``digital_twin_ui.extraction.xplt_parser`` on all .xplt
   files in ``data_path`` to produce a pressure CSV.
2. **build_dataset** — run ``digital_twin_ui.ml.dataset.DatasetBuilder`` to
   build the Parquet training set.
3. **train** — run ``digital_twin_ui.ml.trainer.Trainer`` to train and
   checkpoint the ``PressureMLP`` model.

Usage (when enabled)
---------------------
    import os
    os.environ["USE_AZURE_ML"] = "true"

    from digital_twin_ui.ml.azure_ml_pipeline import build_and_submit_pipeline

    job = build_and_submit_pipeline(
        workspace_config_path="workspace_config.json",
        compute_name="cpu-cluster",
        data_path="azureml://datastores/workspaceblobstore/paths/runs/",
        sql_connection_string="postgresql://...",
    )
    print(job.name)
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

AZURE_ML_ENABLED: bool = os.getenv("USE_AZURE_ML", "false").lower() in (
    "1",
    "true",
    "yes",
)

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

_azure_available = False
try:
    from azure.ai.ml import MLClient, dsl, Input, Output
    from azure.ai.ml.entities import AmlCompute, Environment, CommandComponent
    from azure.identity import DefaultAzureCredential

    _azure_available = True
except ImportError:
    # azure-ai-ml not installed — that is fine.  All functions will return
    # None with a warning when AZURE_ML_ENABLED is True.
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_and_submit_pipeline(
    workspace_config_path: str,
    compute_name: str,
    data_path: str,
    sql_connection_string: str,
) -> Any | None:
    """
    Build and submit the 3-step Azure ML pipeline.

    The pipeline performs:
      1. extract  — parse .xplt files → pressure CSV
      2. build_dataset — pressure CSV → Parquet training set
      3. train    — Parquet → trained PressureMLP checkpoint

    Args:
        workspace_config_path: Path to ``workspace_config.json`` for the
                               Azure ML workspace.
        compute_name:          Name of the compute cluster to submit to.
        data_path:             AzureML datastore URI or local path containing
                               the raw simulation run outputs.
        sql_connection_string: Connection string for the SQL results database
                               (passed to the extract step as an env var).

    Returns:
        The submitted Azure ML pipeline job object, or ``None`` if the module
        is disabled or the Azure SDK is not available.
    """
    if not AZURE_ML_ENABLED:
        print(
            "[azure_ml_pipeline] AZURE_ML_ENABLED=False — "
            "set USE_AZURE_ML=true to activate."
        )
        return None

    if not _azure_available:
        print(
            "[azure_ml_pipeline] azure-ai-ml package is not installed. "
            "Install it with: pip install azure-ai-ml azure-identity"
        )
        return None

    return _submit(
        workspace_config_path=workspace_config_path,
        compute_name=compute_name,
        data_path=data_path,
        sql_connection_string=sql_connection_string,
    )


def get_pipeline_status(job_name: str, workspace_config_path: str) -> Any | None:
    """
    Return the status of a previously submitted pipeline job.

    Args:
        job_name:              Azure ML job name (returned by
                               ``build_and_submit_pipeline``).
        workspace_config_path: Path to ``workspace_config.json``.

    Returns:
        Azure ML Job object or ``None`` if disabled / unavailable.
    """
    if not AZURE_ML_ENABLED:
        print(
            "[azure_ml_pipeline] AZURE_ML_ENABLED=False — "
            "set USE_AZURE_ML=true to activate."
        )
        return None

    if not _azure_available:
        print(
            "[azure_ml_pipeline] azure-ai-ml package is not installed."
        )
        return None

    ml_client = _get_client(workspace_config_path)
    return ml_client.jobs.get(job_name)


# ---------------------------------------------------------------------------
# Internal implementation (only reached when Azure SDK is available)
# ---------------------------------------------------------------------------


def _get_client(workspace_config_path: str) -> "MLClient":
    """Create an MLClient from a workspace config file."""
    return MLClient.from_config(
        credential=DefaultAzureCredential(),
        path=workspace_config_path,
    )


def _submit(
    workspace_config_path: str,
    compute_name: str,
    data_path: str,
    sql_connection_string: str,
) -> Any:
    """Build the DSL pipeline and submit it."""
    ml_client = _get_client(workspace_config_path)

    # ------------------------------------------------------------------
    # Step components
    # ------------------------------------------------------------------

    # Step 1: Extract contact pressure from .xplt files
    extract_component = CommandComponent(
        name="extract_pressure",
        display_name="Extract contact pressure from xplt files",
        description=(
            "Parses all .xplt FEBio result files in the input directory "
            "and writes a pressure_timeseries.csv to the output directory."
        ),
        inputs={
            "runs_dir": Input(type="uri_folder", description="Directory of run folders"),
        },
        outputs={
            "output_dir": Output(type="uri_folder"),
        },
        environment="azureml:dtui-env@latest",
        code=".",
        command=(
            "python -c \""
            "from pathlib import Path; "
            "from digital_twin_ui.extraction.xplt_parser import extract_contact_pressure; "
            "import csv; "
            "runs_dir = Path('${{inputs.runs_dir}}'); "
            "out_dir = Path('${{outputs.output_dir}}'); "
            "out_dir.mkdir(parents=True, exist_ok=True); "
            "rows = []; "
            "[rows.extend([{'run_id': xplt.parent.name, 'time_s': t, "
            "'max_pressure_pa': float(r.max_pressure), 'mean_pressure_pa': float(mp)} "
            "for t, mp in zip(r.times, r.mean_pressure)] "
            "for xplt in runs_dir.rglob('*.xplt') "
            "for r in [extract_contact_pressure(xplt)]); "
            "with open(out_dir / 'pressure_timeseries.csv', 'w') as f: "
            "  w = csv.DictWriter(f, fieldnames=['run_id','time_s','max_pressure_pa','mean_pressure_pa']); "
            "  w.writeheader(); w.writerows(rows)"
            "\""
        ),
    )

    # Step 2: Build training dataset (CSV → Parquet)
    build_dataset_component = CommandComponent(
        name="build_dataset",
        display_name="Build ML training dataset",
        description=(
            "Joins simulation parameters with extracted pressure data "
            "to produce a Parquet training set."
        ),
        inputs={
            "pressure_csv": Input(type="uri_file"),
            "runs_dir": Input(type="uri_folder"),
        },
        outputs={
            "dataset_parquet": Output(type="uri_file"),
        },
        environment="azureml:dtui-env@latest",
        code=".",
        command=(
            "python -m digital_twin_ui.ml.dataset "
            "--pressure-csv ${{inputs.pressure_csv}} "
            "--runs-dir ${{inputs.runs_dir}} "
            "--output ${{outputs.dataset_parquet}}"
        ),
    )

    # Step 3: Train the PressureMLP model
    train_component = CommandComponent(
        name="train_model",
        display_name="Train PressureMLP surrogate model",
        description=(
            "Trains the PressureMLP PyTorch model on the Parquet dataset "
            "and writes the checkpoint to the output directory."
        ),
        inputs={
            "dataset_parquet": Input(type="uri_file"),
        },
        outputs={
            "checkpoint_dir": Output(type="uri_folder"),
        },
        environment="azureml:dtui-env@latest",
        code=".",
        command=(
            "python -m digital_twin_ui.ml.trainer "
            "--dataset ${{inputs.dataset_parquet}} "
            "--checkpoint-dir ${{outputs.checkpoint_dir}}"
        ),
    )

    # ------------------------------------------------------------------
    # DSL Pipeline
    # ------------------------------------------------------------------

    @dsl.pipeline(
        name="dtui_training_pipeline",
        description="Digital Twin UI: extract → build_dataset → train",
        default_compute=compute_name,
    )
    def dtui_pipeline(runs_dir_input: Input):
        extract_step = extract_component(runs_dir=runs_dir_input)
        extract_step.environment_variables = {
            "SQL_CONNECTION_STRING": sql_connection_string,
        }

        build_step = build_dataset_component(
            pressure_csv=extract_step.outputs.output_dir,
            runs_dir=runs_dir_input,
        )

        train_step = train_component(
            dataset_parquet=build_step.outputs.dataset_parquet,
        )

        return {
            "checkpoint_dir": train_step.outputs.checkpoint_dir,
        }

    # Instantiate pipeline with the data path
    pipeline_job = dtui_pipeline(
        runs_dir_input=Input(type="uri_folder", path=data_path),
    )

    # Submit
    submitted_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="dtui_surrogate_training",
    )

    print(
        f"[azure_ml_pipeline] Submitted pipeline job: {submitted_job.name}\n"
        f"  Studio URL: {submitted_job.studio_url}"
    )
    return submitted_job
