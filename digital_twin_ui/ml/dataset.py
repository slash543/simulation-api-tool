"""
Dataset builder for the Digital Twin UI platform.

Reads per-run CSV extractions from ``data/raw/`` and merges them into a
single Parquet dataset suitable for ML training.

Each CSV row represents one simulation run and must have at minimum:
  - ``speed_mm_s``   : float — the insertion speed
  - ``max_pressure`` : float — peak contact pressure

The merged dataset is written to ``data/datasets/catheter_dataset.parquet``.

Usage::

    from digital_twin_ui.ml.dataset import DatasetBuilder

    builder = DatasetBuilder()
    df = builder.build()            # scan raw/ → merge → write parquet → return df
    df = builder.load()             # load existing parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from digital_twin_ui.app.core.config import get_settings
from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Column names (canonical)
# ---------------------------------------------------------------------------

COL_SPEED = "speed_mm_s"
COL_MAX_PRESSURE = "max_pressure"
COL_MEAN_PRESSURE = "mean_pressure"
COL_RUN_ID = "run_id"


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """
    Merge per-run CSV extractions into a single Parquet dataset.

    Args:
        raw_dir: Directory containing per-run CSV files.
        dataset_path: Path where the merged Parquet file is written.
    """

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        dataset_path: Optional[Path] = None,
    ) -> None:
        cfg = get_settings()
        self._raw_dir = raw_dir or (cfg.project_root / "data" / "raw")
        self._dataset_path = dataset_path or cfg.dataset_path_abs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, glob_pattern: str = "*.csv") -> pd.DataFrame:
        """
        Scan ``raw_dir`` for CSV files, merge them, and write Parquet.

        Args:
            glob_pattern: Glob for matching input files (default ``*.csv``).

        Returns:
            Merged DataFrame (may be empty if no files are found).
        """
        csv_files = sorted(self._raw_dir.glob(glob_pattern))
        logger.info("Building dataset", n_files=len(csv_files), raw_dir=str(self._raw_dir))

        if not csv_files:
            logger.warning("No CSV files found in %s", self._raw_dir)
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                frames.append(df)
                logger.debug("Loaded %s (%d rows)", csv_path.name, len(df))
            except Exception as exc:
                logger.warning("Failed to read %s: %s", csv_path.name, exc)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        merged = self._clean(merged)

        self._write(merged)
        logger.info("Dataset written", path=str(self._dataset_path), rows=len(merged))
        return merged

    def load(self) -> pd.DataFrame:
        """
        Load the existing Parquet dataset.

        Returns:
            DataFrame loaded from ``dataset_path``.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
        """
        if not self._dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self._dataset_path}. Run build() first."
            )
        df = pd.read_parquet(self._dataset_path)
        logger.info("Dataset loaded", path=str(self._dataset_path), rows=len(df))
        return df

    def append(self, new_rows: pd.DataFrame) -> pd.DataFrame:
        """
        Append new rows to the existing dataset (or create it).

        Args:
            new_rows: DataFrame with same schema as the dataset.

        Returns:
            Updated combined DataFrame.
        """
        if self._dataset_path.exists():
            existing = pd.read_parquet(self._dataset_path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows.copy()

        combined = self._clean(combined)
        self._write(combined)
        logger.info("Dataset appended", path=str(self._dataset_path), total_rows=len(combined))
        return combined

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows missing required columns and deduplicate."""
        required = [COL_SPEED, COL_MAX_PRESSURE]
        for col in required:
            if col not in df.columns:
                logger.warning("Required column '%s' missing — skipping cleaning", col)
                return df

        before = len(df)
        df = df.dropna(subset=required)
        df = df[df[COL_SPEED] > 0]
        df = df.reset_index(drop=True)

        if COL_RUN_ID in df.columns:
            # Only deduplicate rows that actually have a run_id (not NaN)
            mask = df[COL_RUN_ID].notna()
            deduped = df[mask].drop_duplicates(subset=[COL_RUN_ID])
            df = pd.concat([deduped, df[~mask]], ignore_index=True)

        after = len(df)
        if before != after:
            logger.debug("Cleaned %d → %d rows", before, after)
        return df

    def _write(self, df: pd.DataFrame) -> None:
        """Write DataFrame to Parquet, creating parent directories as needed."""
        self._dataset_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._dataset_path, index=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def raw_dir(self) -> Path:
        return self._raw_dir

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path


# ---------------------------------------------------------------------------
# PressureResult → DataFrame row helper
# ---------------------------------------------------------------------------

def pressure_result_to_row(
    run_id: str,
    speed_mm_s: float,
    pressure_result: Any,
) -> pd.DataFrame:
    """
    Convert a :class:`~digital_twin_ui.extraction.xplt_parser.PressureResult`
    (or its dict form) into a single-row DataFrame.

    Args:
        run_id: Unique identifier for this simulation run.
        speed_mm_s: Insertion speed.
        pressure_result: ``PressureResult`` object or its ``as_dict()`` output.

    Returns:
        Single-row DataFrame.
    """
    if hasattr(pressure_result, "as_dict"):
        d = pressure_result.as_dict()
    else:
        d = dict(pressure_result)

    row: dict = {
        COL_RUN_ID: run_id,
        COL_SPEED: speed_mm_s,
        COL_MAX_PRESSURE: float(d.get("max_pressure", 0.0)),
        COL_MEAN_PRESSURE: float(np.mean(d.get("mean_pressure", [0.0]))),
        "n_faces": int(d.get("n_faces", 0)),
        "variable_name": str(d.get("variable_name", "")),
    }
    return pd.DataFrame([row])


# Avoid circular import at module level
from typing import Any  # noqa: E402 (needed after pressure_result_to_row def)
