"""
Celery task definitions for the Digital Twin UI platform.

Tasks
-----
``run_simulation_task``
    Configure and execute one FEBio simulation for a given speed.
    Returns the serialised :class:`~digital_twin_ui.simulation.simulation_runner.RunResult`.

``run_doe_campaign_task``
    Execute a full Design-of-Experiments campaign (N simulations) sequentially
    inside a single Celery worker.

``extract_results_task``
    Parse an ``.xplt`` result file and return a serialised
    :class:`~digital_twin_ui.extraction.xplt_parser.PressureResult`.

``log_simulation_to_mlflow_task``
    Log a completed simulation result to MLflow.

``run_full_pipeline_task``
    Orchestrate the full pipeline: configure → simulate → extract → MLflow.

All task return values are plain JSON-serialisable dicts so they are compatible
with any Celery result backend.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from celery import Task
from celery.utils.log import get_task_logger

from digital_twin_ui.tasks.celery_app import celery_app

logger = get_task_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_runner():
    from digital_twin_ui.simulation.simulation_runner import SimulationRunner
    return SimulationRunner()


def _import_extractor():
    from digital_twin_ui.extraction.xplt_parser import extract_contact_pressure
    return extract_contact_pressure


def _import_mlflow_manager():
    from digital_twin_ui.experiments.mlflow_manager import MLflowManager
    return MLflowManager()


# ---------------------------------------------------------------------------
# Base task with common error handling
# ---------------------------------------------------------------------------

class _BaseTask(Task):
    """Base task that captures and re-raises exceptions with structured logging."""

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(
            "Task %s[%s] failed: %s",
            self.name,
            task_id,
            exc,
        )
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning(
            "Task %s[%s] retrying: %s",
            self.name,
            task_id,
            exc,
        )
        super().on_retry(exc, task_id, args, kwargs, einfo)

    def on_success(self, retval, task_id, args, kwargs):
        logger.info("Task %s[%s] succeeded", self.name, task_id)
        super().on_success(retval, task_id, args, kwargs)


# ---------------------------------------------------------------------------
# Task: run a single simulation
# ---------------------------------------------------------------------------

@celery_app.task(
    base=_BaseTask,
    name="digital_twin_ui.tasks.run_simulation",
    bind=True,
    max_retries=0,  # simulations are long; do not retry automatically
    time_limit=4000,  # hard kill limit (> timeout_seconds in settings)
    soft_time_limit=3700,
)
def run_simulation_task(
    self: Task,
    speed_mm_s: float,
    run_id: str | None = None,
    template: str = "sample_catheterization",
    speeds_mm_s: list[float] | None = None,
    dwell_time_s: float = 1.0,
) -> dict[str, Any]:
    """
    Run one FEBio catheter simulation.

    Args:
        speed_mm_s:   Insertion speed in mm/s (used for single-step templates).
        run_id:       Optional explicit run identifier.
        template:     Template name (default: ``"sample_catheterization"``).
        speeds_mm_s:  Per-step speeds for multi-step templates.
        dwell_time_s: Dwell time per step in seconds.

    Returns:
        JSON-serialisable dict from :meth:`RunResult.as_dict`.
    """
    logger.info(
        "Starting simulation task: speed=%.3f mm/s, template=%s, run_id=%s",
        speed_mm_s,
        template,
        run_id,
    )
    runner = _import_runner()
    result = runner.run(
        speed_mm_s=speed_mm_s,
        run_id=run_id,
        template=template,
        speeds_mm_s=speeds_mm_s,
        dwell_time_s=dwell_time_s,
    )
    logger.info(
        "Simulation task finished: run_id=%s status=%s duration=%.1fs",
        result.run_id,
        result.status.value,
        result.duration_s or 0.0,
    )
    return result.as_dict()


# ---------------------------------------------------------------------------
# Task: extract results from an xplt file
# ---------------------------------------------------------------------------

@celery_app.task(
    base=_BaseTask,
    name="digital_twin_ui.tasks.extract_results",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
)
def extract_results_task(
    self: Task,
    xplt_path: str,
    variable_name: str = "contact pressure",
) -> dict[str, Any]:
    """
    Parse an ``.xplt`` result file and extract contact pressure.

    Args:
        xplt_path: Absolute path to the ``.xplt`` file.
        variable_name: Surface variable name to extract.

    Returns:
        JSON-serialisable dict from :meth:`PressureResult.as_dict`.
    """
    logger.info("Extracting results from %s", xplt_path)
    extract_fn = _import_extractor()
    try:
        result = extract_fn(Path(xplt_path), variable_name=variable_name)
    except Exception as exc:
        logger.error("Extraction failed: %s\n%s", exc, traceback.format_exc())
        raise self.retry(exc=exc)
    return result.as_dict()


# ---------------------------------------------------------------------------
# Task: log a simulation result to MLflow
# ---------------------------------------------------------------------------

@celery_app.task(
    base=_BaseTask,
    name="digital_twin_ui.tasks.log_to_mlflow",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
)
def log_to_mlflow_task(
    self: Task,
    run_name: str,
    speed_mm_s: float,
    pressure_result: dict[str, Any],
    artefact_path: str | None = None,
    extra_params: dict[str, Any] | None = None,
) -> str:
    """
    Log a completed simulation result to MLflow.

    Args:
        run_name: Human-readable MLflow run name.
        speed_mm_s: Insertion speed used.
        pressure_result: Dict from :meth:`PressureResult.as_dict`.
        artefact_path: Optional CSV/Parquet file path to attach.
        extra_params: Additional parameters to log.

    Returns:
        MLflow run ID string.
    """
    logger.info("Logging simulation to MLflow: run_name=%s speed=%.3f", run_name, speed_mm_s)
    try:
        mgr = _import_mlflow_manager()
        mlflow_run_id = mgr.log_simulation_run(
            run_name=run_name,
            speed_mm_s=speed_mm_s,
            max_pressure=pressure_result.get("max_pressure", 0.0),
            mean_pressures=pressure_result.get("mean_pressure", []),
            times=pressure_result.get("times", []),
            artefact_path=Path(artefact_path) if artefact_path else None,
            extra_params=extra_params,
        )
    except Exception as exc:
        logger.error("MLflow logging failed: %s", exc)
        raise self.retry(exc=exc)
    return mlflow_run_id


# ---------------------------------------------------------------------------
# Task: run a DOE campaign
# ---------------------------------------------------------------------------

@celery_app.task(
    base=_BaseTask,
    name="digital_twin_ui.tasks.run_doe_campaign",
    bind=True,
    max_retries=0,
    time_limit=None,  # unbounded — N simulations can take many hours
)
def run_doe_campaign_task(
    self: Task,
    n_samples: int = 10,
    speed_min: float = 10.0,
    speed_max: float = 25.0,
    sampler: str = "lhs",
    seed: int | None = None,
    extract: bool = True,
    log_mlflow: bool = False,
    template: str = "DT_BT_14Fr_FO_10E_IR12",
    max_perturbation: float = 0.20,
    dwell_time_s: float = 1.0,
) -> dict[str, Any]:
    """
    Run a full DOE campaign: sample speeds, simulate each, optionally extract.

    For multi-step templates, uses :class:`CorrelatedSpeedSampler` to generate
    a ``(n_samples, n_steps)`` speed matrix.  For ``sample_catheterization``,
    uses the existing 1-D scalar sampler.

    Args:
        n_samples:        Number of simulation samples.
        speed_min:        Minimum speed in mm/s.
        speed_max:        Maximum speed in mm/s.
        sampler:          Sampling strategy for single-step templates
                          (``"lhs"``, ``"sobol"``, ``"uniform"``).
        seed:             RNG seed for reproducibility.
        extract:          Whether to run xplt extraction after each simulation.
        log_mlflow:       Whether to log each run to MLflow.
        template:         Template name.  Determines single-step vs multi-step.
        max_perturbation: Fractional perturbation for CorrelatedSpeedSampler.
        dwell_time_s:     Dwell time per step in seconds.

    Returns:
        Dict with ``"samples"`` key listing per-simulation result dicts.
    """
    logger.info(
        "Starting DOE campaign: n=%d speed=[%.1f, %.1f] sampler=%s template=%s",
        n_samples,
        speed_min,
        speed_max,
        sampler,
        template,
    )

    runner = _import_runner()
    extract_fn = _import_extractor() if extract else None
    mgr = _import_mlflow_manager() if log_mlflow else None

    samples: list[dict[str, Any]] = []

    if template == "sample_catheterization":
        # --- Single-step path (backwards compatible) ---
        from digital_twin_ui.doe.sampler import get_sampler

        sampler_obj = get_sampler(sampler)
        scalar_speeds = sampler_obj.sample(n_samples, speed_min, speed_max, seed=seed).tolist()

        for i, speed in enumerate(scalar_speeds):
            logger.info("DOE sample %d/%d: speed=%.3f mm/s", i + 1, n_samples, speed)
            run_result = runner.run(speed_mm_s=float(speed), template=template)
            entry: dict[str, Any] = {"simulation": run_result.as_dict()}

            if extract and extract_fn is not None and run_result.succeeded and run_result.xplt_file.exists():
                try:
                    pressure = extract_fn(run_result.xplt_file)
                    entry["extraction"] = pressure.as_dict()

                    if log_mlflow and mgr is not None:
                        mlflow_id = mgr.log_simulation_run(
                            run_name=run_result.run_id,
                            speed_mm_s=float(speed),
                            max_pressure=pressure.max_pressure,
                            mean_pressures=list(pressure.mean_pressure),
                            times=list(pressure.times),
                        )
                        entry["mlflow_run_id"] = mlflow_id
                except Exception as exc:
                    logger.warning("Extraction failed for %s: %s", run_result.run_id, exc)
                    entry["extraction_error"] = str(exc)

            samples.append(entry)

    else:
        # --- Multi-step path ---
        from digital_twin_ui.doe.correlated_sampler import CorrelatedSpeedSampler
        from digital_twin_ui.simulation.template_registry import get_template_registry

        registry = get_template_registry()
        tc = registry.get(template)

        corr_sampler = CorrelatedSpeedSampler(max_perturbation=max_perturbation)
        speed_matrix = corr_sampler.sample(
            n_samples=n_samples,
            speed_min=speed_min,
            speed_max=speed_max,
            n_steps=tc.n_steps,
            seed=seed,
        )  # shape (n_samples, n_steps)

        for i, speed_row in enumerate(speed_matrix):
            step_speeds = speed_row.tolist()
            mean_speed = float(speed_row.mean())
            logger.info(
                "DOE sample %d/%d: mean_speed=%.3f mm/s",
                i + 1,
                n_samples,
                mean_speed,
            )
            run_result = runner.run(
                speed_mm_s=mean_speed,
                template=template,
                speeds_mm_s=step_speeds,
                dwell_time_s=dwell_time_s,
            )
            entry = {"simulation": run_result.as_dict()}

            if extract and extract_fn is not None and run_result.succeeded and run_result.xplt_file.exists():
                try:
                    pressure = extract_fn(run_result.xplt_file)
                    entry["extraction"] = pressure.as_dict()

                    if log_mlflow and mgr is not None:
                        mlflow_id = mgr.log_simulation_run(
                            run_name=run_result.run_id,
                            speed_mm_s=mean_speed,
                            max_pressure=pressure.max_pressure,
                            mean_pressures=list(pressure.mean_pressure),
                            times=list(pressure.times),
                        )
                        entry["mlflow_run_id"] = mlflow_id
                except Exception as exc:
                    logger.warning("Extraction failed for %s: %s", run_result.run_id, exc)
                    entry["extraction_error"] = str(exc)

            samples.append(entry)

    n_ok = sum(s["simulation"]["status"] == "COMPLETED" for s in samples)
    logger.info("DOE campaign complete: %d/%d succeeded", n_ok, n_samples)
    return {"n_samples": n_samples, "samples": samples}


# ---------------------------------------------------------------------------
# Task: full pipeline for one speed
# ---------------------------------------------------------------------------

@celery_app.task(
    base=_BaseTask,
    name="digital_twin_ui.tasks.run_full_pipeline",
    bind=True,
    max_retries=0,
    time_limit=4200,
)
def run_full_pipeline_task(
    self: Task,
    speed_mm_s: float,
    run_id: str | None = None,
    log_mlflow: bool = True,
    variable_name: str = "contact pressure",
    template: str = "sample_catheterization",
    speeds_mm_s: list[float] | None = None,
    dwell_time_s: float = 1.0,
) -> dict[str, Any]:
    """
    Full pipeline: configure → simulate → extract contact pressure → MLflow.

    Args:
        speed_mm_s:   Insertion speed in mm/s (single-step templates).
        run_id:       Optional explicit run identifier.
        log_mlflow:   Whether to log to MLflow.
        variable_name: xplt surface variable name to extract.
        template:     Template name.
        speeds_mm_s:  Per-step speeds for multi-step templates.
        dwell_time_s: Dwell time per step in seconds.

    Returns:
        Dict with keys ``"simulation"``, ``"extraction"``, and optionally
        ``"mlflow_run_id"``.
    """
    logger.info("Full pipeline: speed=%.3f mm/s, template=%s", speed_mm_s, template)

    # 1. Simulate
    runner = _import_runner()
    sim_result = runner.run(
        speed_mm_s=speed_mm_s,
        run_id=run_id,
        template=template,
        speeds_mm_s=speeds_mm_s,
        dwell_time_s=dwell_time_s,
    )
    output: dict[str, Any] = {"simulation": sim_result.as_dict()}

    if not sim_result.succeeded:
        logger.warning("Simulation did not succeed — skipping extraction")
        return output

    # 2. Extract
    extract_fn = _import_extractor()
    if not sim_result.xplt_file.exists():
        logger.warning("xplt file not found: %s", sim_result.xplt_file)
        output["extraction_error"] = "xplt file not found"
        return output

    try:
        pressure = extract_fn(sim_result.xplt_file, variable_name=variable_name)
        output["extraction"] = pressure.as_dict()
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        output["extraction_error"] = str(exc)
        return output

    # 3. Log to MLflow
    if log_mlflow:
        try:
            mgr = _import_mlflow_manager()
            mlflow_run_id = mgr.log_simulation_run(
                run_name=sim_result.run_id,
                speed_mm_s=speed_mm_s,
                max_pressure=pressure.max_pressure,
                mean_pressures=list(pressure.mean_pressure),
                times=list(pressure.times),
            )
            output["mlflow_run_id"] = mlflow_run_id
        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)
            output["mlflow_error"] = str(exc)

    return output
