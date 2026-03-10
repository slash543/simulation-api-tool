"""
XPLT Parser
===========
Reads FEBio 4.x binary result files (.xplt) and extracts simulation data,
with a focus on contact pressure from surface variables.

Binary format (FEBio xplt, little-endian):
  - Magic: 4 bytes  ``0x00464542``
  - Chunks: ``tag (uint32) | size (uint32) | content (size bytes)``
  - Certain chunks are containers (nested sub-chunks); the rest are leaf
    chunks holding raw data.

Data layout inside each state variable block (``PLT_DATA_VALUES``):
  ``uint32  n_regions``   (0 = global / nodal; ≥1 = region-indexed)
  ``uint32  total_bytes`` (total byte count of the float array that follows)
  ``float32[total_bytes/4]``  flat array of values

Usage
-----
    from pathlib import Path
    from digital_twin_ui.extraction.xplt_parser import extract_contact_pressure

    result = extract_contact_pressure(Path("runs/run_001/results.xplt"))
    print(result.times)            # array of simulation times
    print(result.max_pressure)    # max contact pressure per state
    print(result.mean_pressure)   # mean contact pressure per state
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# File-format constants
# ---------------------------------------------------------------------------

XPLT_MAGIC: int = 0x00464542

# --- Header ------------------------------------------------------------------
PLT_ROOT          = 0x01000000
PLT_HEADER_INFO   = 0x01010000
PLT_HDR_VERSION   = 0x01010001   # uint32
PLT_HDR_COMPRESS  = 0x01010004   # uint32
PLT_HDR_SOFTWARE  = 0x01010006   # char[N]

# --- Dictionary --------------------------------------------------------------
PLT_DICTIONARY    = 0x01020000
PLT_DIC_NODAL     = 0x01023000   # section for nodal variables
PLT_DIC_DOMAIN    = 0x01024000   # section for domain/element variables
PLT_DIC_SURFACE   = 0x01025000   # section for surface variables
PLT_DIC_ITEM      = 0x01020001   # one variable entry
PLT_DIC_ITEM_TYPE = 0x01020002   # uint32: data type
PLT_DIC_ITEM_FMT  = 0x01020003   # uint32: storage format
PLT_DIC_ITEM_NAME = 0x01020004   # char[64]: variable name
PLT_DIC_ITEM_ASIZE = 0x01020005  # uint32: array size (for array type)

# --- Mesh -------------------------------------------------------------------
PLT_MESH          = 0x01040000
PLT_NODE_SECT     = 0x01041000
PLT_NODE_HDR      = 0x01041100
PLT_NODE_SIZE     = 0x01041101   # uint32: number of nodes
PLT_NODE_DIM      = 0x01041102   # uint32: spatial dimension
PLT_NODE_COORDS   = 0x01041200   # float32 array: node coordinates
PLT_SURFACE_SECT  = 0x01043000
PLT_SURFACE       = 0x01043100
PLT_SURF_HDR      = 0x01043101   # surface header (contains sub-chunks)
PLT_SURF_ID       = 0x01043102   # uint32: surface ID
PLT_SURF_NFACES   = 0x01043103   # uint32: number of faces
PLT_SURF_NAME     = 0x01043104   # char[N]: surface name
PLT_SURF_FACETYPE = 0x01043105   # uint32: nodes per face
PLT_SURF_FACES    = 0x01043200   # face connectivity data

# --- States -----------------------------------------------------------------
PLT_STATE         = 0x02000000
PLT_STATE_HDR     = 0x02010000
PLT_STATE_TIME    = 0x02010002   # float32: simulation time
PLT_STATE_ID      = 0x02010003   # uint32: state index
PLT_STATE_DATA    = 0x02020000
PLT_NODE_DATA_SECT  = 0x02020300
PLT_ELEM_DATA_SECT  = 0x02020400
PLT_SURF_DATA_SECT  = 0x02020500   # surface variable data
PLT_DATA_ITEM       = 0x02020001   # per-variable block
PLT_DATA_VAR_IDX    = 0x02020002   # uint32: variable index (1-based)
PLT_DATA_VALUES     = 0x02020003   # binary: values (see module docstring)

# Dictionary item type codes (what kind of tensor)
DICT_TYPE_SCALAR = 0   # 1 float per item
DICT_TYPE_VEC3F  = 1   # 3 floats per item
DICT_TYPE_MAT3FS = 2   # 6 floats per item (symmetric tensor)
DICT_TYPE_MAT3FD = 3   # 3 floats per item (diagonal tensor)
DICT_TYPE_TENS4FS = 4  # 21 floats per item

_COMPONENTS_PER_TYPE: dict[int, int] = {
    DICT_TYPE_SCALAR: 1,
    DICT_TYPE_VEC3F:  3,
    DICT_TYPE_MAT3FS: 6,
    DICT_TYPE_MAT3FD: 3,
    DICT_TYPE_TENS4FS: 21,
}

# Dictionary section tag → variable category name
_DICT_SECTION_NAME: dict[int, str] = {
    PLT_DIC_NODAL:   "nodal",
    PLT_DIC_DOMAIN:  "domain",
    PLT_DIC_SURFACE: "surface",
}

# Chunk tags that are containers (hold sub-chunks)
_CONTAINERS: frozenset[int] = frozenset({
    PLT_ROOT,
    PLT_HEADER_INFO,
    PLT_DICTIONARY,
    PLT_DIC_NODAL, PLT_DIC_DOMAIN, PLT_DIC_SURFACE,
    PLT_DIC_ITEM,
    PLT_MESH,
    PLT_NODE_SECT, PLT_NODE_HDR,
    PLT_SURFACE_SECT, PLT_SURFACE, PLT_SURF_HDR,
    PLT_STATE, PLT_STATE_HDR,
    PLT_STATE_DATA,
    PLT_NODE_DATA_SECT, PLT_ELEM_DATA_SECT, PLT_SURF_DATA_SECT,
    PLT_DATA_ITEM,
})


# ---------------------------------------------------------------------------
# Low-level chunk I/O
# ---------------------------------------------------------------------------


def _iter_chunks(buf: bytes, start: int, end: int):
    """
    Yield ``(tag: int, content: bytes)`` for each chunk in ``buf[start:end]``.

    Silently stops on truncated data.
    """
    off = start
    while off + 8 <= end:
        tag, size = struct.unpack_from("<II", buf, off)
        content_end = off + 8 + size
        yield tag, buf[off + 8 : content_end]
        off = content_end


def _find(buf: bytes, start: int, end: int, target: int) -> bytes | None:
    """Return the content of the first chunk with *target* tag, or ``None``."""
    for tag, content in _iter_chunks(buf, start, end):
        if tag == target:
            return content
    return None


def _read_uint32(buf: bytes) -> int:
    return struct.unpack_from("<I", buf)[0]


def _read_float32(buf: bytes) -> float:
    return struct.unpack_from("<f", buf)[0]


def _read_string(buf: bytes) -> str:
    """
    Decode a null-terminated byte buffer as UTF-8 (with ASCII fallback).

    Stops at the first null byte so that extra bytes after the terminator
    (as written by some FEBio versions) are not included.
    """
    null_pos = buf.find(b"\x00")
    raw = buf[:null_pos] if null_pos != -1 else buf
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("ascii", errors="replace")


def _read_lstring(buf: bytes) -> str:
    """
    Decode a length-prefixed string: ``uint32 length`` + ``char[length]``.

    Used by FEBio for software name and surface name chunks.  Falls back to
    null-terminated if the buffer does not look length-prefixed.
    """
    if len(buf) < 4:
        return _read_string(buf)
    length = struct.unpack_from("<I", buf, 0)[0]
    # Sanity check: length must fit in the buffer
    if 0 < length <= len(buf) - 4:
        raw = buf[4 : 4 + length].rstrip(b"\x00")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("ascii", errors="replace")
    # Fallback: treat as plain null-terminated string
    return _read_string(buf)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DictVariable:
    """One entry from the xplt dictionary."""

    name: str
    data_type: int       # scalar / vec3 / mat3s / ...
    fmt: int             # FMT_NODE=0 / FMT_ITEM=1 / FMT_REGION=3
    section: str         # "nodal" | "domain" | "surface"
    index_in_section: int  # 0-based position within its dictionary section

    @property
    def n_components(self) -> int:
        return _COMPONENTS_PER_TYPE.get(self.data_type, 1)


@dataclass
class SurfacePatch:
    """Geometry of one surface (set of faces) from the mesh section."""

    surface_id: int
    name: str
    n_faces: int
    nodes_per_face: int
    face_connectivity: np.ndarray | None = None  # shape (n_faces, nodes_per_face), 0-based node indices


@dataclass
class StateVariableData:
    """
    Extracted data for one variable in one simulation state.

    ``values`` is a flat float32 array:
      - nodal variable : shape ``(n_nodes * n_components,)``
      - surface variable : shape ``(n_total_faces * n_components,)``
    """

    var_index: int        # 1-based index matching dictionary order
    values: np.ndarray
    per_region: dict[int, np.ndarray] = field(default_factory=dict)  # region_id -> float32 array


@dataclass
class SimulationState:
    """All data for one solver time step."""

    state_index: int
    time: float
    surface_data: list[StateVariableData] = field(default_factory=list)
    node_data: list[StateVariableData] = field(default_factory=list)
    element_data: list[StateVariableData] = field(default_factory=list)


@dataclass
class XpltData:
    """
    Complete structured data extracted from a single ``.xplt`` file.

    Attributes:
        version:      File format version integer (e.g. 53 for FEBio 4.10).
        software:     Software string from the header.
        n_nodes:      Number of mesh nodes.
        surfaces:     List of :class:`SurfacePatch` from the mesh.
        nodal_vars:   Nodal variable entries from the dictionary.
        domain_vars:  Domain/element variable entries from the dictionary.
        surface_vars: Surface variable entries from the dictionary.
        states:       Ordered list of :class:`SimulationState`.
        node_coords:  Node coordinates, shape (n_nodes, 3), float32 xyz in mm.
    """

    version: int = 0
    software: str = ""
    n_nodes: int = 0
    surfaces: list[SurfacePatch] = field(default_factory=list)
    nodal_vars: list[DictVariable] = field(default_factory=list)
    domain_vars: list[DictVariable] = field(default_factory=list)
    surface_vars: list[DictVariable] = field(default_factory=list)
    states: list[SimulationState] = field(default_factory=list)
    node_coords: np.ndarray | None = None  # shape (n_nodes, 3), float32 xyz coordinates

    @property
    def times(self) -> np.ndarray:
        return np.array([s.time for s in self.states], dtype=np.float64)

    @property
    def n_surface_faces(self) -> int:
        return sum(s.n_faces for s in self.surfaces)

    def surface_by_id(self, surface_id: int) -> SurfacePatch | None:
        return next((s for s in self.surfaces if s.surface_id == surface_id), None)

    def surface_by_name(self, name: str) -> SurfacePatch | None:
        return next((s for s in self.surfaces if s.name == name), None)

    def compute_facet_areas(self, surface_name: str) -> np.ndarray:
        """Compute triangle area for each face in the named surface.
        Returns float64 array of shape (n_faces,) in mm²."""
        surf = self.surface_by_name(surface_name)
        if surf is None:
            raise ValueError(f"Surface '{surface_name}' not found")
        if surf.face_connectivity is None:
            raise ValueError(f"Face connectivity not available for '{surface_name}'")
        if self.node_coords is None:
            raise ValueError("Node coordinates not available")
        connectivity = surf.face_connectivity  # (n_faces, nodes_per_face)
        coords = self.node_coords.astype(np.float64)  # (n_nodes, 3)
        n_faces = surf.n_faces
        areas = np.zeros(n_faces, dtype=np.float64)
        for i in range(n_faces):
            face_nodes = connectivity[i]
            p0 = coords[face_nodes[0]]
            p1 = coords[face_nodes[1]]
            p2 = coords[face_nodes[2]]
            ab = p1 - p0
            ac = p2 - p0
            cross = np.cross(ab, ac)
            areas[i] = 0.5 * np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
            if surf.nodes_per_face == 4:
                # Quad: add second triangle
                p3 = coords[face_nodes[3]]
                ad = p3 - p0
                cross2 = np.cross(ac, ad)
                areas[i] += 0.5 * np.sqrt(cross2[0]**2 + cross2[1]**2 + cross2[2]**2)
        return areas


@dataclass(frozen=True)
class PressureResult:
    """
    Contact pressure time series extracted from an xplt file.

    Attributes:
        times:          Simulation times (seconds), shape ``(n_states,)``.
        max_pressure:   Maximum contact pressure per state, ``(n_states,)``.
        mean_pressure:  Mean contact pressure per state, ``(n_states,)``.
        pressures:      Per-face pressure arrays, list of length ``n_states``.
                        Each entry has shape ``(n_faces,)``.
        n_faces:        Total face count across all contact surfaces.
        variable_name:  Name of the surface variable that was extracted.
        source_path:    Path to the ``.xplt`` file.
    """

    times: np.ndarray
    max_pressure: np.ndarray
    mean_pressure: np.ndarray
    pressures: list[np.ndarray]
    n_faces: int
    variable_name: str
    source_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "variable_name": self.variable_name,
            "n_states": len(self.times),
            "n_faces": self.n_faces,
            "times": self.times.tolist(),
            "max_pressure": self.max_pressure.tolist(),
            "mean_pressure": self.mean_pressure.tolist(),
        }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class XpltParser:
    """
    Reads a FEBio ``.xplt`` file and returns structured :class:`XpltData`.

    The parser is stateless — each :meth:`parse` call is independent.
    """

    def parse(self, xplt_path: Path) -> XpltData:
        """
        Parse *xplt_path* and return an :class:`XpltData` instance.

        Args:
            xplt_path: Path to the ``.xplt`` result file.

        Returns:
            :class:`XpltData`

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the magic number is wrong (not an xplt file).
        """
        if not xplt_path.exists():
            raise FileNotFoundError(f"XPLT file not found: {xplt_path}")

        buf = xplt_path.read_bytes()
        if len(buf) < 4:
            raise ValueError(f"File too small to be a valid xplt: {xplt_path}")

        magic = struct.unpack_from("<I", buf, 0)[0]
        if magic != XPLT_MAGIC:
            raise ValueError(
                f"Not a valid xplt file (magic=0x{magic:08X}, "
                f"expected 0x{XPLT_MAGIC:08X}): {xplt_path}"
            )

        data = XpltData()
        data.version, data.software = self._parse_header(buf)
        data.nodal_vars, data.domain_vars, data.surface_vars = self._parse_dictionary(buf)
        data.surfaces, data.n_nodes, data.node_coords = self._parse_mesh(buf)
        data.states = self._parse_states(buf, data.surface_vars)

        logger.info(
            "XPLT parsed",
            path=str(xplt_path),
            version=data.version,
            n_nodes=data.n_nodes,
            n_surfaces=len(data.surfaces),
            n_surface_vars=len(data.surface_vars),
            n_states=len(data.states),
        )
        return data

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _parse_header(self, buf: bytes) -> tuple[int, str]:
        """Return (version, software_string)."""
        root = _find(buf, 4, len(buf), PLT_ROOT)
        if root is None:
            return 0, ""
        hdr = _find(root, 0, len(root), PLT_HEADER_INFO)
        if hdr is None:
            return 0, ""

        version = 0
        software = ""
        for tag, content in _iter_chunks(hdr, 0, len(hdr)):
            if tag == PLT_HDR_VERSION and len(content) >= 4:
                version = _read_uint32(content)
            elif tag == PLT_HDR_SOFTWARE and content:
                software = _read_lstring(content)

        return version, software

    # ------------------------------------------------------------------
    # Dictionary
    # ------------------------------------------------------------------

    def _parse_dictionary(
        self, buf: bytes
    ) -> tuple[list[DictVariable], list[DictVariable], list[DictVariable]]:
        """Return (nodal_vars, domain_vars, surface_vars)."""
        root = _find(buf, 4, len(buf), PLT_ROOT)
        if root is None:
            return [], [], []
        dictionary = _find(root, 0, len(root), PLT_DICTIONARY)
        if dictionary is None:
            return [], [], []

        nodal_vars: list[DictVariable] = []
        domain_vars: list[DictVariable] = []
        surface_vars: list[DictVariable] = []

        section_targets = {
            PLT_DIC_NODAL:   nodal_vars,
            PLT_DIC_DOMAIN:  domain_vars,
            PLT_DIC_SURFACE: surface_vars,
        }

        for sec_tag, sec_content in _iter_chunks(dictionary, 0, len(dictionary)):
            target_list = section_targets.get(sec_tag)
            if target_list is None:
                continue
            section_name = _DICT_SECTION_NAME[sec_tag]
            idx = 0
            for item_tag, item_content in _iter_chunks(sec_content, 0, len(sec_content)):
                if item_tag != PLT_DIC_ITEM:
                    continue
                name, data_type, fmt = "", 0, 0
                for t, c in _iter_chunks(item_content, 0, len(item_content)):
                    if t == PLT_DIC_ITEM_NAME and c:
                        name = _read_string(c[:64])
                    elif t == PLT_DIC_ITEM_TYPE and len(c) >= 4:
                        data_type = _read_uint32(c)
                    elif t == PLT_DIC_ITEM_FMT and len(c) >= 4:
                        fmt = _read_uint32(c)
                var = DictVariable(
                    name=name,
                    data_type=data_type,
                    fmt=fmt,
                    section=section_name,
                    index_in_section=idx,
                )
                target_list.append(var)
                idx += 1

        return nodal_vars, domain_vars, surface_vars

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------

    def _parse_mesh(self, buf: bytes) -> tuple[list[SurfacePatch], int, np.ndarray | None]:
        """Return (surfaces, n_nodes, node_coords)."""
        mesh = _find(buf, 4, len(buf), PLT_MESH)
        if mesh is None:
            return [], 0, None

        n_nodes = self._parse_n_nodes(mesh)
        node_coords = self._parse_node_coords(mesh)
        surfaces = self._parse_surfaces(mesh)
        return surfaces, n_nodes, node_coords

    def _parse_node_coords(self, mesh_buf: bytes) -> np.ndarray | None:
        """Parse node coordinates from PLT_NODE_COORDS.
        Returns array of shape (n_nodes, 3) with float32 xyz coordinates.
        The raw buffer stores [node_id(uint32), x(f32), y(f32), z(f32)] per node.
        """
        node_sect = _find(mesh_buf, 0, len(mesh_buf), PLT_NODE_SECT)
        if node_sect is None:
            return None
        coords_raw = _find(node_sect, 0, len(node_sect), PLT_NODE_COORDS)
        if coords_raw is None or len(coords_raw) < 16:
            return None
        # Each node: [node_id(uint32 but skip), x(f32), y(f32), z(f32)]
        n_values = len(coords_raw) // 4
        n_nodes = n_values // 4  # 4 values per node
        if n_nodes == 0:
            return None
        try:
            data_f32 = np.frombuffer(coords_raw, dtype=np.float32).reshape(n_nodes, 4)
        except ValueError:
            return None
        return data_f32[:, 1:].copy()  # skip column 0 (node_id), keep xyz

    def _parse_n_nodes(self, mesh_buf: bytes) -> int:
        node_sect = _find(mesh_buf, 0, len(mesh_buf), PLT_NODE_SECT)
        if node_sect is None:
            return 0
        node_hdr = _find(node_sect, 0, len(node_sect), PLT_NODE_HDR)
        if node_hdr is None:
            return 0
        size_bytes = _find(node_hdr, 0, len(node_hdr), PLT_NODE_SIZE)
        if size_bytes is None or len(size_bytes) < 4:
            return 0
        return _read_uint32(size_bytes)

    def _parse_surfaces(self, mesh_buf: bytes) -> list[SurfacePatch]:
        surf_sect = _find(mesh_buf, 0, len(mesh_buf), PLT_SURFACE_SECT)
        if surf_sect is None:
            return []

        surfaces: list[SurfacePatch] = []
        for tag, content in _iter_chunks(surf_sect, 0, len(surf_sect)):
            if tag != PLT_SURFACE:
                continue
            hdr_bytes = _find(content, 0, len(content), PLT_SURF_HDR)
            if hdr_bytes is None:
                continue

            surf_id = 0
            n_faces = 0
            name = ""
            nodes_per_face = 0
            for t, c in _iter_chunks(hdr_bytes, 0, len(hdr_bytes)):
                if t == PLT_SURF_ID and len(c) >= 4:
                    surf_id = _read_uint32(c)
                elif t == PLT_SURF_NFACES and len(c) >= 4:
                    n_faces = _read_uint32(c)
                elif t == PLT_SURF_NAME and c:
                    name = _read_lstring(c)
                elif t == PLT_SURF_FACETYPE and len(c) >= 4:
                    nodes_per_face = _read_uint32(c)

            if surf_id > 0 or n_faces > 0:
                face_data = _find(content, 0, len(content), PLT_SURF_FACES)
                if face_data is not None and n_faces > 0 and nodes_per_face > 0:
                    connectivity = self._parse_face_connectivity(face_data, n_faces, nodes_per_face)
                else:
                    connectivity = None
                surfaces.append(SurfacePatch(
                    surface_id=surf_id,
                    name=name,
                    n_faces=n_faces,
                    nodes_per_face=nodes_per_face,
                    face_connectivity=connectivity,
                ))

        return surfaces

    def _parse_face_connectivity(
        self, face_data: bytes, n_faces: int, nodes_per_face: int
    ) -> np.ndarray | None:
        """Parse face connectivity from PLT_SURF_FACES content.
        Each face is stored as a sub-chunk: tag(4)+size(4)+[face_id, n_nodes, node0..nodeN].
        Returns array of shape (n_faces, nodes_per_face) with 0-based node indices.
        """
        result = np.zeros((n_faces, nodes_per_face), dtype=np.int32)
        face_count = 0
        off = 0
        while off + 8 <= len(face_data) and face_count < n_faces:
            size = struct.unpack_from("<I", face_data, off + 4)[0]
            content_start = off + 8
            content_end = content_start + size
            if content_end > len(face_data):
                break
            content = face_data[content_start:content_end]
            n_ints = size // 4
            if n_ints < 2 + nodes_per_face:
                off = content_end
                continue
            vals = struct.unpack_from(f"<{n_ints}I", content)
            # vals[0] = face_id (1-based, skip)
            # vals[1] = nodes_per_face
            # vals[2:2+nodes_per_face] = 0-based node indices
            actual_nn = vals[1]
            for j in range(min(nodes_per_face, actual_nn)):
                result[face_count, j] = vals[2 + j]
            face_count += 1
            off = content_end
        if face_count == 0:
            return None
        return result

    # ------------------------------------------------------------------
    # States
    # ------------------------------------------------------------------

    def _parse_states(
        self, buf: bytes, surface_vars: list[DictVariable]
    ) -> list[SimulationState]:
        """Parse all simulation states from the file."""
        states: list[SimulationState] = []
        state_idx = 0

        for tag, content in _iter_chunks(buf, 4, len(buf)):
            if tag != PLT_STATE:
                continue

            time = self._read_state_time(content)
            surf_data = self._read_surface_data(content)
            node_data = self._read_section_data(content, PLT_NODE_DATA_SECT)
            elem_data = self._read_section_data(content, PLT_ELEM_DATA_SECT)

            states.append(SimulationState(
                state_index=state_idx,
                time=time,
                surface_data=surf_data,
                node_data=node_data,
                element_data=elem_data,
            ))
            state_idx += 1

        return states

    def _read_state_time(self, state_buf: bytes) -> float:
        hdr = _find(state_buf, 0, len(state_buf), PLT_STATE_HDR)
        if hdr is None:
            return 0.0
        time_bytes = _find(hdr, 0, len(hdr), PLT_STATE_TIME)
        if time_bytes is None or len(time_bytes) < 4:
            return 0.0
        return float(_read_float32(time_bytes))

    def _read_surface_data(self, state_buf: bytes) -> list[StateVariableData]:
        return self._read_section_data(state_buf, PLT_SURF_DATA_SECT)

    def _read_section_data(
        self, state_buf: bytes, section_tag: int
    ) -> list[StateVariableData]:
        """Read all variable blocks from a given section within a state."""
        section = _find(state_buf, 0, len(state_buf), PLT_STATE_DATA)
        if section is None:
            return []
        sect_content = _find(section, 0, len(section), section_tag)
        if sect_content is None:
            return []

        result: list[StateVariableData] = []
        for tag, content in _iter_chunks(sect_content, 0, len(sect_content)):
            if tag != PLT_DATA_ITEM:
                continue
            var_idx_bytes = _find(content, 0, len(content), PLT_DATA_VAR_IDX)
            values_bytes = _find(content, 0, len(content), PLT_DATA_VALUES)
            if var_idx_bytes is None or values_bytes is None:
                continue

            var_index = _read_uint32(var_idx_bytes)
            values, per_region = self._decode_values_with_regions(values_bytes)
            result.append(StateVariableData(var_index=var_index, values=values, per_region=per_region))

        return result

    def _decode_values_with_regions(self, raw: bytes) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Like _decode_values but also returns per-region dict."""
        if len(raw) < 8:
            return np.array([], dtype=np.float32), {}
        parts: list[np.ndarray] = []
        per_region: dict[int, np.ndarray] = {}
        off = 0
        while off + 8 <= len(raw):
            region_id = struct.unpack_from("<I", raw, off)[0]
            byte_count = struct.unpack_from("<I", raw, off + 4)[0]
            if byte_count == 0 or byte_count % 4 != 0:
                break
            end = off + 8 + byte_count
            if end > len(raw):
                break
            arr = np.frombuffer(raw[off + 8: end], dtype=np.float32).copy()
            parts.append(arr)
            per_region[region_id] = arr
            off = end
        flat = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        return flat, per_region

    def _decode_values(self, raw: bytes) -> np.ndarray:
        """
        Decode the payload of a ``PLT_DATA_VALUES`` chunk.

        Layout — repeated until end of buffer:
            uint32   region_id    (surface / domain / node-set identifier)
            uint32   byte_count   (byte length of the float32 array that follows)
            float32[byte_count/4] values for this region

        All regions are concatenated into a single flat array.
        """
        return self._decode_values_with_regions(raw)[0]


# ---------------------------------------------------------------------------
# Contact-pressure extraction
# ---------------------------------------------------------------------------


def _find_pressure_variable(
    surface_vars: list[DictVariable],
    preferred_name: str = "contact pressure",
) -> DictVariable | None:
    """
    Return the best-matching surface variable for contact pressure.

    Matching is case-insensitive.  Falls back to the first surface scalar
    variable if the preferred name is not found.
    """
    target = preferred_name.lower()
    # Exact match first
    for v in surface_vars:
        if v.name.lower() == target:
            return v
    # Partial match
    for v in surface_vars:
        if target in v.name.lower() or "pressure" in v.name.lower():
            return v
    # Any scalar surface variable as last resort
    for v in surface_vars:
        if v.data_type == DICT_TYPE_SCALAR:
            return v
    return None


def extract_contact_pressure(
    xplt_path: Path,
    variable_name: str = "contact pressure",
) -> PressureResult:
    """
    Parse an xplt file and extract contact pressure as a time series.

    The function finds the surface variable whose name best matches
    *variable_name* (case-insensitive), then collects per-face scalar
    pressure values from every simulation state.

    Args:
        xplt_path:     Path to the ``.xplt`` result file.
        variable_name: Name of the surface variable to extract.

    Returns:
        :class:`PressureResult`

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no surface variables are found in the file.
    """
    xplt_path = Path(xplt_path)
    parser = XpltParser()
    xplt_data = parser.parse(xplt_path)

    if not xplt_data.surface_vars:
        raise ValueError(
            f"No surface variables found in {xplt_path}.  "
            "Ensure the simulation is configured to output contact pressure."
        )

    pv = _find_pressure_variable(xplt_data.surface_vars, variable_name)
    if pv is None:
        raise ValueError(
            f"No matching surface variable found for '{variable_name}' "
            f"in {xplt_path}.  Available: "
            f"{[v.name for v in xplt_data.surface_vars]}"
        )

    # The variable index in state data blocks is 1-based within its section
    target_index = pv.index_in_section + 1

    times_list: list[float] = []
    pressures_list: list[np.ndarray] = []

    for state in xplt_data.states:
        times_list.append(state.time)
        # Find the matching variable block
        matched: np.ndarray | None = None
        for sd in state.surface_data:
            if sd.var_index == target_index:
                matched = sd.values
                break
        if matched is None or len(matched) == 0:
            n_faces = max(xplt_data.n_surface_faces, 1)
            matched = np.zeros(n_faces, dtype=np.float32)
        pressures_list.append(matched)

    times = np.array(times_list, dtype=np.float64)

    # Compute per-state summaries (clamp negatives to 0 for contact pressure)
    max_p = np.array(
        [float(np.max(p)) if len(p) > 0 else 0.0 for p in pressures_list],
        dtype=np.float64,
    )
    mean_p = np.array(
        [float(np.mean(p[p > 0])) if np.any(p > 0) else 0.0 for p in pressures_list],
        dtype=np.float64,
    )

    # n_faces: derive from actual data length (the contact variable may not
    # cover every surface in the mesh, e.g. zero-displacement surfaces are
    # excluded from contact output).
    n_faces_actual = len(pressures_list[0]) if pressures_list else xplt_data.n_surface_faces

    result = PressureResult(
        times=times,
        max_pressure=max_p,
        mean_pressure=mean_p,
        pressures=pressures_list,
        n_faces=n_faces_actual,
        variable_name=pv.name,
        source_path=xplt_path,
    )

    logger.info(
        "Contact pressure extracted",
        path=str(xplt_path),
        variable=pv.name,
        n_states=len(times),
        n_faces=xplt_data.n_surface_faces,
        peak_pressure=float(np.max(max_p)) if len(max_p) > 0 else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def parse_xplt(xplt_path: Path) -> XpltData:
    """Parse an xplt file and return the full :class:`XpltData` structure."""
    return XpltParser().parse(Path(xplt_path))


def _decode_values_from_bytes(raw: bytes) -> np.ndarray:
    """
    Module-level wrapper around :meth:`XpltParser._decode_values`.

    Exposed for unit testing without constructing a full parser.
    """
    return XpltParser()._decode_values(raw)
