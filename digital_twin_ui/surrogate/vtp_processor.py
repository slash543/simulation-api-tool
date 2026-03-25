"""
VTP (VTK PolyData XML) file processing for surrogate-model annotation.

Reads VTP files written by xplt_core.export_vtp(), predicts contact pressure
using the surrogate model, and writes new VTP files with the predicted values.

VTP files are self-contained XML with base64-encoded binary arrays — no VTK
library dependency is needed.

Public API
----------
VTPProcessor.read(path)                       -> VTPData
VTPProcessor.predict_and_save(vtp, predictor, depth, out_path)
VTPData.to_facets_df()                        -> DataFrame
compute_csar_from_vtp(vtp, predictor, depths, z_bands) -> DataFrame
"""
from __future__ import annotations

import base64
import logging
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VTPData dataclass
# ---------------------------------------------------------------------------

@dataclass
class VTPData:
    """
    Contents of a single VTP file (one timestep of a simulation surface).

    Attributes
    ----------
    points:
        Node coordinates, shape ``(n_nodes, 3)``, float32 [mm].
    connectivity:
        Flat node-index array for all faces (3 entries per triangle), int32.
    offsets:
        Cumulative offset into connectivity for each face, int32.
        offsets[i] = 3*(i+1) for triangles.
    face_ids:
        Per-face integer IDs (1-based), int32.
    areas:
        Per-face surface areas [mm²], float32.
    contact_pressure:
        Per-face contact pressure [MPa] from FEM (may be zeros for predicted VTPs).
        float32.
    source_path:
        Path of the file this was read from (for traceability).
    """

    points: np.ndarray          # (N, 3) float32
    connectivity: np.ndarray    # (3*F,) int32
    offsets: np.ndarray         # (F,)   int32
    face_ids: np.ndarray        # (F,)   int32
    areas: np.ndarray           # (F,)   float32
    contact_pressure: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    source_path: Path | None = None

    @property
    def n_faces(self) -> int:
        return len(self.face_ids)

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    def centroids(self) -> np.ndarray:
        """Compute face centroids as mean of 3 triangle nodes, shape (F, 3)."""
        n_faces = self.n_faces
        c = np.zeros((n_faces, 3), dtype=np.float64)
        for i in range(n_faces):
            start = 3 * i
            n0, n1, n2 = self.connectivity[start], self.connectivity[start + 1], self.connectivity[start + 2]
            c[i] = (self.points[n0] + self.points[n1] + self.points[n2]) / 3.0
        return c

    def to_facets_df(self) -> pd.DataFrame:
        """
        Convert to a facets DataFrame compatible with the surrogate predictor.

        Returns
        -------
        DataFrame with columns: face_id, centroid_x, centroid_y, centroid_z, facet_area.
        """
        c = self.centroids()
        return pd.DataFrame(
            {
                "face_id": self.face_ids.tolist(),
                "centroid_x": c[:, 0].tolist(),
                "centroid_y": c[:, 1].tolist(),
                "centroid_z": c[:, 2].tolist(),
                "facet_area": self.areas.tolist(),
            }
        )


# ---------------------------------------------------------------------------
# Binary / base64 helpers
# ---------------------------------------------------------------------------

def _decode_b64(encoded: str, dtype: np.dtype) -> np.ndarray:
    """
    Decode a VTK base64 block (4-byte uint32 length header + data).

    VTK inline binary arrays start with a 4-byte little-endian uint32 giving
    the byte count of the payload, followed by the payload itself.
    """
    raw = base64.b64decode(encoded)
    if len(raw) < 4:
        return np.array([], dtype=dtype)
    byte_count = struct.unpack_from("<I", raw, 0)[0]
    payload = raw[4 : 4 + byte_count]
    return np.frombuffer(payload, dtype=dtype).copy()


def _encode_b64(arr: np.ndarray) -> str:
    """Encode a numpy array as VTK inline base64 binary (with 4-byte length header)."""
    raw = arr.tobytes()
    header = struct.pack("<I", len(raw))
    return base64.b64encode(header + raw).decode("ascii")


# ---------------------------------------------------------------------------
# VTPProcessor
# ---------------------------------------------------------------------------

class VTPProcessor:
    """Read, annotate, and write VTP files (VTK PolyData XML format)."""

    @staticmethod
    def read(path: str | Path) -> VTPData:
        """
        Parse a VTP file and return a :class:`VTPData` object.

        Supports VTK PolyData files with inline binary base64 encoding —
        the format written by ``xplt_core.export_vtp()``.

        Parameters
        ----------
        path:
            Path to the ``.vtp`` file.

        Returns
        -------
        VTPData with points, connectivity, offsets, face_ids, areas,
        and contact_pressure arrays.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"VTP file not found: {path}")

        tree = ET.parse(path)
        root = tree.getroot()

        # Locate PolyData / Piece
        poly = root.find(".//PolyData/Piece")
        if poly is None:
            raise ValueError(f"No PolyData/Piece found in {path}")

        # ---------- Points ----------
        pts_data = poly.find(".//Points/DataArray")
        if pts_data is None or pts_data.text is None:
            raise ValueError(f"No Points DataArray in {path}")
        raw_pts = _decode_b64(pts_data.text.strip(), np.float32)
        n_pts = int(poly.get("NumberOfPoints", 0))
        points = raw_pts.reshape(n_pts, 3)

        # ---------- Connectivity + offsets ----------
        polys = poly.find("Polys")
        if polys is None:
            raise ValueError(f"No Polys element in {path}")

        conn_arr = None
        off_arr = None
        for da in polys.findall("DataArray"):
            name = da.get("Name", "")
            if name == "connectivity" and da.text:
                conn_arr = _decode_b64(da.text.strip(), np.int32)
            elif name == "offsets" and da.text:
                off_arr = _decode_b64(da.text.strip(), np.int32)

        if conn_arr is None or off_arr is None:
            raise ValueError(f"Missing connectivity or offsets DataArray in {path}")

        # ---------- Cell data ----------
        face_ids = np.array([], dtype=np.int32)
        areas = np.array([], dtype=np.float32)
        cp = np.array([], dtype=np.float32)

        cell_data = poly.find("CellData")
        if cell_data is not None:
            for da in cell_data.findall("DataArray"):
                name = da.get("Name", "")
                if da.text is None:
                    continue
                text = da.text.strip()
                if name == "face_id":
                    face_ids = _decode_b64(text, np.int32)
                elif name == "area_mm2":
                    areas = _decode_b64(text, np.float32)
                elif name == "contact_pressure_MPa":
                    cp = _decode_b64(text, np.float32)

        n_faces = int(poly.get("NumberOfPolys", len(off_arr)))
        if len(face_ids) == 0:
            face_ids = np.arange(1, n_faces + 1, dtype=np.int32)

        logger.debug(
            "Read VTP %s: %d nodes, %d faces", path.name, len(points), n_faces
        )

        return VTPData(
            points=points,
            connectivity=conn_arr,
            offsets=off_arr,
            face_ids=face_ids,
            areas=areas,
            contact_pressure=cp if len(cp) == n_faces else np.zeros(n_faces, dtype=np.float32),
            source_path=path,
        )

    @staticmethod
    def write(
        path: str | Path,
        vtp: VTPData,
        contact_pressure: np.ndarray | None = None,
    ) -> Path:
        """
        Write a VTP file with (optionally updated) contact-pressure values.

        Parameters
        ----------
        path:
            Output file path.
        vtp:
            Source :class:`VTPData` providing geometry.
        contact_pressure:
            Per-face pressure array [MPa].  If *None*, uses ``vtp.contact_pressure``.

        Returns
        -------
        Absolute path to the written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cp = contact_pressure if contact_pressure is not None else vtp.contact_pressure
        n_faces = vtp.n_faces
        n_pts = vtp.n_nodes

        cp_arr = np.asarray(cp, dtype=np.float32)
        if len(cp_arr) != n_faces:
            raise ValueError(
                f"contact_pressure length {len(cp_arr)} != n_faces {n_faces}"
            )

        pts_flat = vtp.points.astype(np.float32).ravel()

        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" '
            'header_type="UInt32" compressor="">',
            "  <PolyData>",
            f'    <Piece NumberOfPoints="{n_pts}" NumberOfPolys="{n_faces}">',
            "      <Points>",
            '        <DataArray type="Float32" NumberOfComponents="3" '
            f'format="binary">{_encode_b64(pts_flat)}</DataArray>',
            "      </Points>",
            "      <Polys>",
            '        <DataArray type="Int32" Name="connectivity" '
            f'format="binary">{_encode_b64(vtp.connectivity.astype(np.int32))}</DataArray>',
            '        <DataArray type="Int32" Name="offsets" '
            f'format="binary">{_encode_b64(vtp.offsets.astype(np.int32))}</DataArray>',
            "      </Polys>",
            "      <CellData>",
            '        <DataArray type="Int32" Name="face_id" '
            f'format="binary">{_encode_b64(vtp.face_ids.astype(np.int32))}</DataArray>',
        ]

        if len(vtp.areas) == n_faces:
            lines.append(
                '        <DataArray type="Float32" Name="area_mm2" '
                f'format="binary">{_encode_b64(vtp.areas.astype(np.float32))}</DataArray>'
            )

        lines += [
            '        <DataArray type="Float32" Name="contact_pressure_MPa" '
            f'format="binary">{_encode_b64(cp_arr)}</DataArray>',
            "      </CellData>",
            "    </Piece>",
            "  </PolyData>",
            "</VTKFile>",
        ]

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote VTP to %s (%d faces)", path, n_faces)
        return path.resolve()

    @staticmethod
    def predict_and_save(
        vtp: VTPData,
        predictor: Any,  # SurrogatePredictor — avoid circular import
        insertion_depth_mm: float,
        output_path: str | Path,
    ) -> Path:
        """
        Predict contact pressure on *vtp* geometry and save a new VTP file.

        Parameters
        ----------
        vtp:
            Geometry to annotate (from :meth:`read`).
        predictor:
            Loaded :class:`~digital_twin_ui.surrogate.predictor.SurrogatePredictor`.
        insertion_depth_mm:
            Catheter insertion depth [mm].
        output_path:
            Where to write the new VTP file.

        Returns
        -------
        Resolved Path to the written file.
        """
        facets_df = vtp.to_facets_df()
        cp = predictor.predict_at_depth(facets_df, insertion_depth_mm)
        return VTPProcessor.write(output_path, vtp, contact_pressure=cp)


# ---------------------------------------------------------------------------
# CSAR from VTP file
# ---------------------------------------------------------------------------

def compute_csar_from_vtp(
    vtp: VTPData,
    predictor: Any,  # SurrogatePredictor
    insertion_depths: list[float],
    z_bands: list[dict[str, Any]],
    cp_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Compute CSAR vs insertion depth using the geometry from a VTP file.

    Parameters
    ----------
    vtp:
        Geometry parsed by :func:`VTPProcessor.read`.
    predictor:
        Loaded surrogate predictor.
    insertion_depths:
        Insertion depth sample points [mm].
    z_bands:
        List of dicts with ``zmin``, ``zmax``, and optional ``label``.
    cp_threshold:
        Facets with predicted cp > threshold are counted as in contact.

    Returns
    -------
    DataFrame with the same schema as
    :meth:`CSAREngine.compute_csar_vs_depth`.
    """
    from .csar_engine import CSAREngine  # local import to avoid circular

    facets_df = vtp.to_facets_df()
    engine = CSAREngine(predictor, cp_threshold=cp_threshold)
    return engine.compute_csar_vs_depth(facets_df, insertion_depths, z_bands)
