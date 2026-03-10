"""
DOE Pipeline
============
Orchestrates a Design of Experiments campaign by:

  1. Generating parameter samples via a :class:`BaseSampler`.
  2. Running each sample through the simulation engine.
  3. Collecting results into a structured :class:`DOEResult`.

The pipeline is intentionally decoupled from the sampler and runner so that
each component can be tested and replaced independently.

Usage
-----
    from digital_twin_ui.doe.doe_pipeline import DOEPipeline

    pipeline = DOEPipeline()

    # Run a 10-point LHS campaign
    doe_result = pipeline.run(
        n_samples=10,
        sampler_name="lhs",
        seed=42,
    )

    print(doe_result.completed)   # number of successful runs
    print(doe_result.failed)      # number of failed runs
    for r in doe_result.run_results:
        print(r.speed_mm_s, r.status)

    # Export to CSV
    df = doe_result.to_dataframe()
    df.to_csv("doe_results.csv", index=False)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from digital_twin_ui.app.core.config import Settings, get_settings
from digital_twin_ui.app.core.logging import get_logger
from digital_twin_ui.doe.sampler import BaseSampler, SamplerName, get_sampler
from digital_twin_ui.simulation.simulation_runner import RunResult, SimulationRunner, SimulationStatus

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DOEResult:
    """
    Aggregated results from a completed DOE campaign.

    Attributes:
        sampler_name:   Name of the sampler used (e.g. ``"lhs"``).
        n_requested:    Number of samples requested.
        seed:           Random seed used (or None).
        speed_samples:  Sorted array of speed values (mm/s) that were run.
        run_results:    List of :class:`RunResult` objects, one per sample.
        wall_time_s:    Total wall-clock time for the whole campaign.
    """

    sampler_name: str
    n_requested: int
    seed: int | None
    speed_samples: np.ndarray
    run_results: list[RunResult] = field(default_factory=list)
    wall_time_s: float = 0.0

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def completed(self) -> int:
        """Number of runs that completed successfully."""
        return sum(1 for r in self.run_results if r.status == SimulationStatus.COMPLETED)

    @property
    def failed(self) -> int:
        """Number of runs that failed or timed out."""
        return sum(1 for r in self.run_results if r.status != SimulationStatus.COMPLETED)

    @property
    def success_rate(self) -> float:
        """Fraction of runs that completed (0.0 – 1.0)."""
        total = len(self.run_results)
        return self.completed / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dataframe(self):  # type: ignore[return]
        """
        Convert results to a :class:`pandas.DataFrame`.

        Each row corresponds to one simulation run.  Columns include
        ``run_id``, ``speed_mm_s``, ``status``, ``duration_s``,
        ``lc_end_time``, ``time_steps_step2``, ``run_dir``.

        Returns:
            pandas.DataFrame
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pandas is required for to_dataframe()") from exc

        rows: list[dict[str, Any]] = []
        for r in self.run_results:
            rows.append(
                {
                    "run_id": r.run_id,
                    "speed_mm_s": r.speed_mm_s,
                    "status": r.status.value,
                    "duration_s": r.duration_s,
                    "lc_end_time": r.lc_end_time,
                    "time_steps_step2": r.time_steps_step2,
                    "run_dir": str(r.run_dir),
                    "exit_code": r.exit_code,
                    "error_message": r.error_message,
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary dictionary."""
        return {
            "sampler_name": self.sampler_name,
            "n_requested": self.n_requested,
            "n_completed": self.completed,
            "n_failed": self.failed,
            "success_rate": round(self.success_rate, 4),
            "wall_time_s": round(self.wall_time_s, 3),
            "seed": self.seed,
            "speed_range": {
                "min": float(self.speed_samples.min()) if len(self.speed_samples) else None,
                "max": float(self.speed_samples.max()) if len(self.speed_samples) else None,
            },
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DOEPipeline:
    """
    Orchestrates a full DOE campaign.

    Args:
        settings:  Application settings (optional).  Defaults to singleton.
        runner:    Pre-configured :class:`SimulationRunner` (optional).
                   Useful for injecting mocks in tests.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        runner: SimulationRunner | None = None,
    ) -> None:
        self._cfg = settings or get_settings()
        self._runner = runner or SimulationRunner(settings=self._cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        n_samples: int | None = None,
        sampler_name: SamplerName | str | None = None,
        speed_min: float | None = None,
        speed_max: float | None = None,
        seed: int | None = None,
        sampler: BaseSampler | None = None,
        template_name: str | None = None,
        max_perturbation: float | None = None,
        dwell_time_s: float | None = None,
    ) -> DOEResult:
        """
        Execute a synchronous DOE campaign.

        Samples are generated first, then simulations are run sequentially.
        For concurrent execution, use the async Celery tasks instead.

        For multi-step templates (n_steps > 1), uses :class:`CorrelatedSpeedSampler`
        to generate a ``(n_samples, n_steps)`` speed matrix.  For the
        ``sample_catheterization`` single-step template, uses the existing
        1-D scalar sampler.

        Args:
            n_samples:        Number of simulation runs.  Defaults to
                              ``settings.doe.default_num_samples``.
            sampler_name:     ``"lhs"``, ``"sobol"``, or ``"uniform"``.
                              Defaults to ``settings.doe.default_sampler``.
            speed_min:        Lower speed bound (mm/s).
                              Defaults to ``settings.doe.speed_min_mm_s``.
            speed_max:        Upper speed bound (mm/s).
                              Defaults to ``settings.doe.speed_max_mm_s``.
            seed:             Random seed for reproducible sampling.
            sampler:          Pre-built sampler instance (overrides *sampler_name*).
                              Only used for single-step templates.
            template_name:    Template name.  Defaults to
                              ``settings.doe.default_template`` if available,
                              otherwise ``"sample_catheterization"``.
            max_perturbation: Fractional perturbation for CorrelatedSpeedSampler.
                              Defaults to ``settings.doe.max_perturbation`` if
                              available, otherwise 0.20.
            dwell_time_s:     Dwell time per step.  Defaults to
                              ``settings.doe.default_dwell_time_s`` if available,
                              otherwise 1.0.

        Returns:
            :class:`DOEResult` with all run results and summary statistics.
        """
        # Resolve defaults from config
        n_samples = n_samples if n_samples is not None else self._cfg.doe.default_num_samples
        sampler_name = sampler_name or self._cfg.doe.default_sampler
        speed_min = speed_min if speed_min is not None else self._cfg.doe.speed_min_mm_s
        speed_max = speed_max if speed_max is not None else self._cfg.doe.speed_max_mm_s

        # Resolve template with backwards-compat default
        if template_name is None:
            template_name = getattr(
                self._cfg.doe, "default_template", "sample_catheterization"
            )

        # Resolve max_perturbation
        if max_perturbation is None:
            max_perturbation = float(
                getattr(self._cfg.doe, "max_perturbation", 0.20)
            )

        # Resolve dwell_time_s
        if dwell_time_s is None:
            dwell_time_s = float(
                getattr(self._cfg.doe, "default_dwell_time_s", 1.0)
            )

        if template_name == "sample_catheterization":
            # --- Single-step path (backwards compatible) ---
            active_sampler = sampler or get_sampler(sampler_name)
            speed_samples = active_sampler.sample(
                n_samples=n_samples,
                low=speed_min,
                high=speed_max,
                seed=seed,
            )

            logger.info(
                "DOE campaign started (single-step)",
                sampler=active_sampler.name,
                template=template_name,
                n_samples=n_samples,
                speed_min=speed_min,
                speed_max=speed_max,
                seed=seed,
            )

            doe_result = DOEResult(
                sampler_name=active_sampler.name,
                n_requested=n_samples,
                seed=seed,
                speed_samples=speed_samples,
            )

            t_start = time.monotonic()
            run_results: list[RunResult] = []

            for i, speed in enumerate(speed_samples):
                logger.info(
                    "DOE run starting",
                    run_index=i + 1,
                    total=n_samples,
                    speed_mm_s=round(float(speed), 4),
                )
                result = self._runner.run(
                    speed_mm_s=float(speed),
                    template=template_name,
                )
                run_results.append(result)
                logger.info(
                    "DOE run finished",
                    run_index=i + 1,
                    total=n_samples,
                    run_id=result.run_id,
                    status=result.status.value,
                    duration_s=result.duration_s,
                )

        else:
            # --- Multi-step path ---
            from digital_twin_ui.doe.correlated_sampler import CorrelatedSpeedSampler
            from digital_twin_ui.simulation.template_registry import get_template_registry

            registry = get_template_registry()
            tc = registry.get(template_name)

            corr_sampler = CorrelatedSpeedSampler(max_perturbation=max_perturbation)
            speed_matrix = corr_sampler.sample(
                n_samples=n_samples,
                speed_min=speed_min,
                speed_max=speed_max,
                n_steps=tc.n_steps,
                seed=seed,
            )  # shape (n_samples, n_steps)

            # Use the mean speed across steps as the scalar representative
            speed_samples = speed_matrix.mean(axis=1)

            logger.info(
                "DOE campaign started (multi-step)",
                sampler="correlated_lhs",
                template=template_name,
                n_samples=n_samples,
                n_steps=tc.n_steps,
                speed_min=speed_min,
                speed_max=speed_max,
                max_perturbation=max_perturbation,
                dwell_time_s=dwell_time_s,
                seed=seed,
            )

            doe_result = DOEResult(
                sampler_name="correlated_lhs",
                n_requested=n_samples,
                seed=seed,
                speed_samples=speed_samples,
            )

            t_start = time.monotonic()
            run_results = []

            for i, speed_row in enumerate(speed_matrix):
                step_speeds = speed_row.tolist()
                mean_speed = float(speed_row.mean())
                logger.info(
                    "DOE run starting",
                    run_index=i + 1,
                    total=n_samples,
                    mean_speed_mm_s=round(mean_speed, 4),
                )
                result = self._runner.run(
                    speed_mm_s=mean_speed,
                    template=template_name,
                    speeds_mm_s=step_speeds,
                    dwell_time_s=dwell_time_s,
                )
                run_results.append(result)
                logger.info(
                    "DOE run finished",
                    run_index=i + 1,
                    total=n_samples,
                    run_id=result.run_id,
                    status=result.status.value,
                    duration_s=result.duration_s,
                )

        doe_result.run_results = run_results
        doe_result.wall_time_s = time.monotonic() - t_start

        logger.info(
            "DOE campaign complete",
            template=template_name,
            completed=doe_result.completed,
            failed=doe_result.failed,
            wall_time_s=round(doe_result.wall_time_s, 2),
        )

        return doe_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def preview_samples(
        self,
        n_samples: int | None = None,
        sampler_name: SamplerName | str | None = None,
        speed_min: float | None = None,
        speed_max: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Return the speed samples that *would* be used without running anything.

        Useful for inspecting the DOE design before committing to runs.
        """
        n_samples = n_samples if n_samples is not None else self._cfg.doe.default_num_samples
        sampler_name = sampler_name or self._cfg.doe.default_sampler
        speed_min = speed_min if speed_min is not None else self._cfg.doe.speed_min_mm_s
        speed_max = speed_max if speed_max is not None else self._cfg.doe.speed_max_mm_s

        active_sampler = get_sampler(sampler_name)
        return active_sampler.sample(
            n_samples=n_samples,
            low=speed_min,
            high=speed_max,
            seed=seed,
        )
