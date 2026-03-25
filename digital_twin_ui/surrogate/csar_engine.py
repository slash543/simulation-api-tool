"""
Contact Surface Area Ratio (CSAR) engine using surrogate model predictions.

Mirrors the CSAR calculation in xplt_core.py (``SimulationCase.compute_csar``),
but uses the surrogate neural network instead of FEM results.

This allows evaluating CSAR vs insertion depth at arbitrary depths and
insertion speeds without running full FEM simulations.

Public API
----------
CSAREngine(predictor)
engine.compute_csar_vs_depth(facets_df, insertion_depths, z_bands) -> DataFrame
engine.compute_from_csv(csv_path, insertion_depths, z_bands)        -> DataFrame
build_insertion_depths(max_depth, step, start)                       -> list[float]
load_reference_facets(path)                                          -> DataFrame
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .predictor import SurrogatePredictor

logger = logging.getLogger(__name__)

# Columns expected in a reference facets DataFrame
REQUIRED_GEOMETRY_COLS = ["centroid_x", "centroid_y", "centroid_z", "facet_area"]

# Contact-pressure threshold: facets with predicted cp > this value
# are considered "in contact" (matches xplt_core behaviour: cp > 0)
CP_CONTACT_THRESHOLD: float = 0.0


def build_insertion_depths(
    max_depth_mm: float = 300.0,
    step_mm: float = 5.0,
    start_mm: float = 0.0,
) -> list[float]:
    """
    Generate a uniform list of insertion-depth sample points [mm].

    Parameters
    ----------
    max_depth_mm:
        Maximum insertion depth to evaluate.
    step_mm:
        Spacing between sample points.
    start_mm:
        First sample point (inclusive).

    Returns
    -------
    List of floats from *start_mm* to *max_depth_mm* at *step_mm* intervals.
    """
    depths = list(np.arange(start_mm, max_depth_mm + step_mm * 0.5, step_mm))
    return [float(d) for d in depths]


def load_reference_facets(path: str | Path) -> pd.DataFrame:
    """
    Load a reference facets CSV and validate that it has required columns.

    The CSV should contain unique facets (one row per facet, not per timestep).
    If the CSV has an ``insertion_depth`` column it will be dropped — we add it
    per-sample inside CSAREngine.

    Acceptable CSV formats:
    - Direct facets export from ``SimulationCase.df_facets`` (face_id, cx_mm, …)
    - Unique-facet extract from ``SimulationCase.df_surrogate()`` (centroid_x, …)

    Returns
    -------
    DataFrame with at least: centroid_x, centroid_y, centroid_z, facet_area.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference facets file not found: {path}")

    df = pd.read_csv(path)

    # Normalise column names from df_facets format (cx_mm → centroid_x, etc.)
    rename_map = {
        "cx_mm": "centroid_x",
        "cy_mm": "centroid_y",
        "cz_mm": "centroid_z",
        "area_mm2": "facet_area",
    }
    df = df.rename(columns=rename_map)

    missing = [c for c in REQUIRED_GEOMETRY_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Reference facets CSV is missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Drop insertion_depth / contact_pressure if present (we compute them)
    drop_cols = [c for c in ["insertion_depth", "contact_pressure"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Deduplicate rows (in case the CSV is in long format with repeated facets)
    n_before = len(df)
    df = df.drop_duplicates(subset=REQUIRED_GEOMETRY_COLS).reset_index(drop=True)
    if len(df) < n_before:
        logger.info(
            "Deduplicated reference facets: %d → %d unique facets",
            n_before,
            len(df),
        )

    logger.info("Loaded %d reference facets from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# CSAREngine
# ---------------------------------------------------------------------------

class CSAREngine:
    """
    Compute CSAR vs insertion depth using a surrogate model.

    Parameters
    ----------
    predictor:
        A loaded :class:`~digital_twin_ui.surrogate.predictor.SurrogatePredictor`.
    cp_threshold:
        Minimum predicted contact pressure [MPa] for a facet to be counted
        as "in contact".  Default 0.0 matches FEM post-processing behaviour.
    """

    def __init__(
        self,
        predictor: SurrogatePredictor,
        cp_threshold: float = CP_CONTACT_THRESHOLD,
    ) -> None:
        self._predictor = predictor
        self._cp_threshold = cp_threshold

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_csar_vs_depth(
        self,
        facets_df: pd.DataFrame,
        insertion_depths: list[float],
        z_bands: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Compute CSAR for every (insertion_depth, z_band) combination.

        Parameters
        ----------
        facets_df:
            Reference geometry.  Must have centroid_x/y/z and facet_area columns.
            Must NOT include an ``insertion_depth`` column.
        insertion_depths:
            List of insertion depth values [mm] to evaluate.
        z_bands:
            List of Z-band definitions. Each entry must have:
            ``zmin`` (float, mm), ``zmax`` (float, mm).
            Optionally: ``label`` (str, display name).

        Returns
        -------
        DataFrame with columns:
            insertion_depth_mm, zmin_mm, zmax_mm, band_label,
            csar, contact_area_mm2, total_area_mm2,
            n_contact_facets, n_total_facets, mean_cp_MPa, max_cp_MPa.
        """
        if not z_bands:
            raise ValueError("z_bands must be a non-empty list")
        if not insertion_depths:
            raise ValueError("insertion_depths must be a non-empty list")

        missing = [c for c in REQUIRED_GEOMETRY_COLS if c not in facets_df.columns]
        if missing:
            raise ValueError(f"facets_df missing columns: {missing}")

        records: list[dict[str, Any]] = []

        for depth in insertion_depths:
            # Predict contact pressure for all facets at this depth
            cp = self._predictor.predict_at_depth(facets_df, depth)

            for band in z_bands:
                zmin = float(band["zmin"])
                zmax = float(band["zmax"])
                label = str(band.get("label", f"z[{zmin:.0f},{zmax:.0f}]"))

                # Select facets in this Z band
                z = facets_df["centroid_z"].values
                region_mask = (z >= zmin) & (z <= zmax)
                n_total = int(region_mask.sum())

                if n_total == 0:
                    records.append(
                        {
                            "insertion_depth_mm": float(depth),
                            "zmin_mm": zmin,
                            "zmax_mm": zmax,
                            "band_label": label,
                            "csar": float("nan"),
                            "contact_area_mm2": 0.0,
                            "total_area_mm2": 0.0,
                            "n_contact_facets": 0,
                            "n_total_facets": 0,
                            "mean_cp_MPa": float("nan"),
                            "max_cp_MPa": float("nan"),
                        }
                    )
                    continue

                region_areas = facets_df.loc[region_mask, "facet_area"].values
                region_cp = cp[region_mask]

                contact_mask = region_cp > self._cp_threshold
                n_contact = int(contact_mask.sum())
                total_area = float(region_areas.sum())
                contact_area = float(region_areas[contact_mask].sum())
                csar = contact_area / total_area if total_area > 0 else float("nan")

                records.append(
                    {
                        "insertion_depth_mm": float(depth),
                        "zmin_mm": zmin,
                        "zmax_mm": zmax,
                        "band_label": label,
                        "csar": csar,
                        "contact_area_mm2": contact_area,
                        "total_area_mm2": total_area,
                        "n_contact_facets": n_contact,
                        "n_total_facets": n_total,
                        "mean_cp_MPa": float(region_cp.mean()),
                        "max_cp_MPa": float(region_cp.max()),
                    }
                )

        return pd.DataFrame(records)

    def compute_from_csv(
        self,
        csv_path: str | Path,
        insertion_depths: list[float],
        z_bands: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Load reference facets from CSV and compute CSAR vs depth.

        Convenience wrapper around :meth:`compute_csar_vs_depth`.
        """
        facets_df = load_reference_facets(csv_path)
        return self.compute_csar_vs_depth(facets_df, insertion_depths, z_bands)

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def csar_to_dict(df: pd.DataFrame) -> dict[str, Any]:
        """
        Convert the CSAR DataFrame to a JSON-serialisable dict for API responses.

        Groups results by band_label and returns per-label time series.
        """
        result: dict[str, Any] = {
            "insertion_depths_mm": sorted(df["insertion_depth_mm"].unique().tolist()),
            "bands": {},
        }
        for label, group in df.groupby("band_label"):
            g = group.sort_values("insertion_depth_mm")
            result["bands"][str(label)] = {
                "zmin_mm": float(g["zmin_mm"].iloc[0]),
                "zmax_mm": float(g["zmax_mm"].iloc[0]),
                "label": str(label),
                "insertion_depths_mm": g["insertion_depth_mm"].tolist(),
                "csar": [None if np.isnan(v) else float(v) for v in g["csar"]],
                "contact_area_mm2": g["contact_area_mm2"].tolist(),
                "total_area_mm2": float(g["total_area_mm2"].iloc[0]),
                "n_contact_facets": g["n_contact_facets"].tolist(),
                "n_total_facets": int(g["n_total_facets"].iloc[0]),
                "mean_cp_MPa": g["mean_cp_MPa"].tolist(),
                "max_cp_MPa": g["max_cp_MPa"].tolist(),
            }
        return result
