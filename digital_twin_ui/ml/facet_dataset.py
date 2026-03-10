"""
Facet-level dataset builder for MLP training.

Builds a Parquet dataset from multiple simulation runs where each row
represents (facet_id, speed_mm_s) → (facet_area, contact_pressure).

Two modes:
  - Per-timestep: one row per (run, facet, time_step)
  - Peak only: one row per (run, facet) with peak pressure

Usage::

    from digital_twin_ui.ml.facet_dataset import FacetDatasetBuilder

    builder = FacetDatasetBuilder(surface_name="SlidingElastic1Primary")
    df = builder.build([
        {"xplt_path": "runs/run_001/results.xplt", "speed_mm_s": 5.0, "run_id": "run_001"},
        {"xplt_path": "runs/run_002/results.xplt", "speed_mm_s": 4.5, "run_id": "run_002"},
    ])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from digital_twin_ui.app.core.logging import get_logger
from digital_twin_ui.extraction.facet_tracker import FacetTracker

logger = get_logger(__name__)

# Column names
COL_RUN_ID = "run_id"
COL_FACET_ID = "facet_id"
COL_SURFACE_NAME = "surface_name"
COL_SPEED = "speed_mm_s"
COL_AREA = "facet_area"
COL_TIME_STEP = "time_step"
COL_TIME_S = "time_s"
COL_PRESSURE = "contact_pressure"
COL_PEAK_PRESSURE = "peak_contact_pressure"


class FacetDatasetBuilder:
    """
    Build a Parquet training dataset from multiple FEBio simulation runs.

    Args:
        surface_name: Name of the surface to extract data from.
        facet_ids: 0-based facet indices to track (None = all).
        variable_name: Surface variable name to extract.
        peak_only: If True, each row is (facet_id, speed) → (area, peak_pressure).
                   If False, each row is (facet_id, speed, time_step) → (area, pressure).
        dataset_path: Where to save the Parquet file.
    """

    def __init__(
        self,
        surface_name: str = "SlidingElastic1Primary",
        facet_ids: list[int] | None = None,
        variable_name: str = "contact pressure",
        peak_only: bool = True,
        dataset_path: Path | None = None,
    ) -> None:
        self._surface_name = surface_name
        self._facet_ids = facet_ids
        self._variable_name = variable_name
        self._peak_only = peak_only
        self._tracker = FacetTracker(variable_name=variable_name)

        if dataset_path is None:
            from digital_twin_ui.app.core.config import get_settings
            cfg = get_settings()
            self._dataset_path = cfg.project_root / "data" / "datasets" / "facet_dataset.parquet"
        else:
            self._dataset_path = dataset_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        run_configs: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Build dataset from list of run configs.

        Each config dict must have:
          - ``xplt_path``: str or Path
          - ``speed_mm_s``: float
          - ``run_id``: str (optional; defaults to xplt_path stem)

        Args:
            run_configs: List of run configuration dicts.

        Returns:
            Merged DataFrame saved to ``dataset_path``.
        """
        frames: list[pd.DataFrame] = []

        for i, cfg in enumerate(run_configs):
            xplt_path = Path(cfg["xplt_path"])
            speed = float(cfg["speed_mm_s"])
            run_id = str(cfg.get("run_id", xplt_path.stem))

            logger.info(
                "Processing run %d/%d: %s speed=%.2f",
                i + 1, len(run_configs), run_id, speed
            )

            try:
                df_run = self._process_run(xplt_path, speed, run_id)
                frames.append(df_run)
            except Exception as exc:
                logger.warning("Failed to process run %s: %s", run_id, exc)

        if not frames:
            logger.warning("No runs processed — returning empty DataFrame")
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        self._save(merged)
        logger.info(
            "Facet dataset built: %d rows from %d runs → %s",
            len(merged), len(frames), self._dataset_path
        )
        return merged

    def load(self) -> pd.DataFrame:
        """Load existing Parquet dataset."""
        if not self._dataset_path.exists():
            raise FileNotFoundError(
                f"Facet dataset not found: {self._dataset_path}. Run build() first."
            )
        return pd.read_parquet(self._dataset_path)

    def append_run(
        self,
        xplt_path: Path,
        speed_mm_s: float,
        run_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Extract one run and append it to the existing dataset.

        Args:
            xplt_path: Path to the .xplt result file.
            speed_mm_s: Insertion speed for this run.
            run_id: Run identifier (defaults to file stem).

        Returns:
            Updated full dataset.
        """
        run_id = run_id or Path(xplt_path).stem
        df_new = self._process_run(Path(xplt_path), speed_mm_s, run_id)

        if self._dataset_path.exists():
            existing = pd.read_parquet(self._dataset_path)
            # Remove any previous rows for the same run_id
            if COL_RUN_ID in existing.columns:
                existing = existing[existing[COL_RUN_ID] != run_id]
            merged = pd.concat([existing, df_new], ignore_index=True)
        else:
            merged = df_new

        self._save(merged)
        return merged

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_run(
        self, xplt_path: Path, speed_mm_s: float, run_id: str
    ) -> pd.DataFrame:
        """Extract facet data for one run and return a DataFrame."""
        series_list = self._tracker.extract(
            xplt_path=xplt_path,
            speed_mm_s=speed_mm_s,
            surface_name=self._surface_name,
            facet_ids=self._facet_ids,
            variable_name=self._variable_name,
        )

        if self._peak_only:
            rows = [
                {
                    COL_RUN_ID: run_id,
                    COL_FACET_ID: ts.facet_id,
                    COL_SURFACE_NAME: ts.surface_name,
                    COL_SPEED: ts.speed_mm_s,
                    COL_AREA: ts.area,
                    COL_PEAK_PRESSURE: ts.peak_pressure,
                }
                for ts in series_list
            ]
        else:
            rows = []
            for ts in series_list:
                for row in ts.as_rows():
                    row[COL_RUN_ID] = run_id
                    rows.append(row)

        return pd.DataFrame(rows)

    def _save(self, df: pd.DataFrame) -> None:
        self._dataset_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._dataset_path, index=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

    @property
    def surface_name(self) -> str:
        return self._surface_name
