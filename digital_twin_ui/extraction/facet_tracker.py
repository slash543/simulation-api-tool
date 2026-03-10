"""
Facet-level contact pressure tracker.

Extracts per-facet pressure time series from an .xplt file for selected facets.
Computes facet areas from mesh geometry.

Usage::

    from digital_twin_ui.extraction.facet_tracker import FacetTracker

    tracker = FacetTracker()
    series = tracker.extract(
        xplt_path=Path("runs/run_001/results.xplt"),
        speed_mm_s=5.0,
        surface_name="SlidingElastic1Primary",
        facet_ids=[0, 1, 2, 100],   # 0-based within surface; None = all
        variable_name="contact pressure",
    )
    for ts in series:
        print(ts.facet_id, ts.area, ts.pressures.shape)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from digital_twin_ui.app.core.logging import get_logger
from digital_twin_ui.extraction.xplt_parser import XpltData, XpltParser, _find_pressure_variable

logger = get_logger(__name__)


@dataclass
class FacetInfo:
    """Geometry of one tracked facet."""
    facet_id: int        # 0-based local index within the surface
    surface_name: str
    surface_id: int
    area: float          # mm²
    node_indices: list[int]  # 0-based mesh node indices


@dataclass
class FacetTimeSeries:
    """Contact pressure time series for one facet across all simulation states."""
    facet_id: int        # 0-based local index within the surface
    surface_name: str
    surface_id: int
    area: float          # mm²
    speed_mm_s: float
    times: np.ndarray    # shape (n_states,)
    pressures: np.ndarray  # shape (n_states,) - pressure at this facet per time step

    @property
    def peak_pressure(self) -> float:
        return float(np.max(self.pressures)) if len(self.pressures) > 0 else 0.0

    def as_rows(self) -> list[dict[str, Any]]:
        """Return list of dict rows for DataFrame construction (one per time step)."""
        return [
            {
                "facet_id": self.facet_id,
                "surface_name": self.surface_name,
                "surface_id": self.surface_id,
                "speed_mm_s": self.speed_mm_s,
                "facet_area": self.area,
                "time_step": i,
                "time_s": float(self.times[i]),
                "contact_pressure": float(self.pressures[i]),
            }
            for i in range(len(self.times))
        ]


class FacetTracker:
    """
    Extracts per-facet contact pressure time series from an .xplt file.

    Args:
        variable_name: Default surface variable name to extract.
    """

    def __init__(self, variable_name: str = "contact pressure") -> None:
        self._variable_name = variable_name
        self._parser = XpltParser()

    def extract(
        self,
        xplt_path: Path,
        speed_mm_s: float,
        surface_name: str,
        facet_ids: list[int] | None = None,
        variable_name: str | None = None,
    ) -> list[FacetTimeSeries]:
        """
        Extract per-facet pressure time series for selected facets.

        Args:
            xplt_path: Path to the .xplt result file.
            speed_mm_s: Insertion speed used for this simulation run.
            surface_name: Name of the surface to extract data from (e.g. "SlidingElastic1Primary").
            facet_ids: 0-based local facet indices to track (None = all facets).
            variable_name: Override default variable name.

        Returns:
            List of FacetTimeSeries, one per selected facet.
        """
        vname = variable_name or self._variable_name
        xplt_path = Path(xplt_path)
        xplt_data = self._parser.parse(xplt_path)

        # Validate surface
        surf = xplt_data.surface_by_name(surface_name)
        if surf is None:
            available = [s.name for s in xplt_data.surfaces]
            raise ValueError(
                f"Surface '{surface_name}' not found. Available: {available}"
            )

        # Determine facet selection
        if facet_ids is None:
            selected_ids = list(range(surf.n_faces))
        else:
            out_of_range = [i for i in facet_ids if not (0 <= i < surf.n_faces)]
            if out_of_range:
                raise ValueError(
                    f"Facet IDs {out_of_range} out of range [0, {surf.n_faces}) "
                    f"for surface '{surface_name}'"
                )
            selected_ids = list(facet_ids)

        # Compute facet areas
        try:
            areas = xplt_data.compute_facet_areas(surface_name)
        except ValueError as exc:
            logger.warning("Could not compute facet areas: %s — using 0.0", exc)
            areas = np.zeros(surf.n_faces, dtype=np.float64)

        # Face connectivity for node indices
        if surf.face_connectivity is not None:
            connectivity = surf.face_connectivity
        else:
            connectivity = None

        # Find pressure variable
        pv = _find_pressure_variable(xplt_data.surface_vars, vname)
        if pv is None:
            raise ValueError(
                f"No matching surface variable for '{vname}'. "
                f"Available: {[v.name for v in xplt_data.surface_vars]}"
            )
        target_index = pv.index_in_section + 1  # 1-based

        # Extract per-facet pressures across all states
        n_states = len(xplt_data.states)
        n_selected = len(selected_ids)

        pressures_by_facet = np.zeros((n_states, n_selected), dtype=np.float32)
        times = np.array([s.time for s in xplt_data.states], dtype=np.float64)

        for si, state in enumerate(xplt_data.states):
            for sv in state.surface_data:
                if sv.var_index != target_index:
                    continue
                region_data = sv.per_region.get(surf.surface_id)
                if region_data is None:
                    break
                for j, fid in enumerate(selected_ids):
                    if fid < len(region_data):
                        pressures_by_facet[si, j] = region_data[fid]
                break

        # Build FacetTimeSeries list
        result: list[FacetTimeSeries] = []
        for j, fid in enumerate(selected_ids):
            node_indices = (
                connectivity[fid].tolist() if connectivity is not None else []
            )
            result.append(FacetTimeSeries(
                facet_id=fid,
                surface_name=surface_name,
                surface_id=surf.surface_id,
                area=float(areas[fid]),
                speed_mm_s=speed_mm_s,
                times=times.copy(),
                pressures=pressures_by_facet[:, j].copy(),
            ))

        logger.info(
            "Facet extraction complete",
            surface=surface_name,
            n_facets=n_selected,
            n_states=n_states,
            speed=speed_mm_s,
        )
        return result

    def get_facet_info(
        self,
        xplt_path: Path,
        surface_name: str,
        facet_ids: list[int] | None = None,
    ) -> list[FacetInfo]:
        """
        Return geometry information for selected facets without loading state data.

        Args:
            xplt_path: Path to the .xplt file.
            surface_name: Name of the surface.
            facet_ids: 0-based facet indices (None = all).

        Returns:
            List of FacetInfo with area and node indices.
        """
        xplt_data = self._parser.parse(xplt_path)
        surf = xplt_data.surface_by_name(surface_name)
        if surf is None:
            raise ValueError(f"Surface '{surface_name}' not found")

        selected_ids = facet_ids if facet_ids is not None else list(range(surf.n_faces))
        try:
            areas = xplt_data.compute_facet_areas(surface_name)
        except ValueError:
            areas = np.zeros(surf.n_faces, dtype=np.float64)

        result = []
        for fid in selected_ids:
            node_indices = (
                surf.face_connectivity[fid].tolist()
                if surf.face_connectivity is not None else []
            )
            result.append(FacetInfo(
                facet_id=fid,
                surface_name=surface_name,
                surface_id=surf.surface_id,
                area=float(areas[fid]),
                node_indices=node_indices,
            ))
        return result
