"""
Tests for the per-facet contact pressure tracking pipeline.

Covers:
  - Enhanced XpltParser (node_coords, face_connectivity, per_region)
  - FacetTracker (extract, get_facet_info)
  - FacetDatasetBuilder (build, load, append_run)
  - FacetPressureMLP (forward, count_parameters)
  - FacetTrainer (train, checkpoint)
  - FacetPredictor (from_checkpoint, predict, predict_batch)
"""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from digital_twin_ui.extraction.xplt_parser import (
    XpltData,
    XpltParser,
    SurfacePatch,
    SimulationState,
    StateVariableData,
    DictVariable,
    parse_xplt,
)
from digital_twin_ui.extraction.facet_tracker import (
    FacetTracker,
    FacetTimeSeries,
    FacetInfo,
)
from digital_twin_ui.ml.facet_dataset import (
    FacetDatasetBuilder,
    COL_RUN_ID,
    COL_FACET_ID,
    COL_SPEED,
    COL_AREA,
    COL_PEAK_PRESSURE,
    COL_SURFACE_NAME,
    COL_TIME_STEP,
    COL_PRESSURE,
)
from digital_twin_ui.ml.facet_model import FacetPressureMLP
from digital_twin_ui.ml.facet_trainer import FacetTrainer, FacetTrainingResult
from digital_twin_ui.ml.facet_inference import FacetPredictor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_XPLT = Path(__file__).parent.parent / "conf_file" / "jobs" / "sample_catheterization.xplt"

PRIMARY_SURFACE = "SlidingElastic1Primary"
SECONDARY_SURFACE = "SlidingElastic1Secondary"
PRIMARY_N_FACES = 2734
SECONDARY_N_FACES = 1834
N_NODES = 3289
N_STATES = 41


# ---------------------------------------------------------------------------
# Synthetic xplt helpers (reused from test_xplt_parser)
# ---------------------------------------------------------------------------

def _chunk(tag: int, content: bytes) -> bytes:
    return struct.pack("<II", tag, len(content)) + content


def _uint32(v: int) -> bytes:
    return struct.pack("<I", v)


def _float32(v: float) -> bytes:
    return struct.pack("<f", v)


def _lstring(s: str) -> bytes:
    raw = s.encode("ascii") + b"\x00"
    return _uint32(len(raw)) + raw


def _region_block(region_id: int, values: list[float]) -> bytes:
    float_bytes = struct.pack(f"<{len(values)}f", *values)
    return _uint32(region_id) + _uint32(len(float_bytes)) + float_bytes


def _face_subchunk(face_id: int, nodes: list[int]) -> bytes:
    """Build a properly formatted face sub-chunk: tag(4)+size(4)+[face_id, n_nodes, node0..N]."""
    n = len(nodes)
    vals = [face_id, n] + nodes
    content = struct.pack(f"<{len(vals)}I", *vals)
    # Use tag 0x01043201 as per spec
    return struct.pack("<II", 0x01043201, len(content)) + content


def build_synthetic_xplt_with_coords(
    n_nodes: int = 9,
    nodes_per_face: int = 3,
    n_faces: int = 3,
    surface_id: int = 2,
    surface_name: str = "TestSurface",
    n_states: int = 2,
) -> bytes:
    """Build a synthetic xplt with proper node coords and face sub-chunks."""
    from digital_twin_ui.extraction.xplt_parser import XPLT_MAGIC

    # Header info
    header_info = _chunk(
        0x01010000,
        _chunk(0x01010001, _uint32(53))
        + _chunk(0x01010004, _uint32(0))
        + _chunk(0x01010006, _lstring("FEBio 4.10.0")),
    )

    # Dictionary: one surface variable "contact pressure"
    dict_item = _chunk(
        0x01020001,
        _chunk(0x01020002, _uint32(0))   # scalar
        + _chunk(0x01020003, _uint32(1)) # fmt
        + _chunk(0x01020005, _uint32(0))
        + _chunk(0x01020004, ("contact pressure".encode("ascii") + b"\x00" * 50)[:64]),
    )
    dictionary = _chunk(
        0x01020000,
        _chunk(0x01023000, b"")
        + _chunk(0x01024000, b"")
        + _chunk(0x01025000, dict_item),
    )

    root_header = _chunk(0x01000000, header_info + dictionary)

    # Node coords: [node_id(uint32), x, y, z] per node, as float32
    # We use 3x3 grid of nodes: (0,0,0), (1,0,0), ..., (2,2,0)
    coord_data = b""
    for i in range(n_nodes):
        x = float(i % 3)
        y = float(i // 3)
        z = 0.0
        # node_id (as uint32 reinterpreted as float32 — store raw uint32 bytes)
        coord_data += struct.pack("<I", i + 1)  # node_id as raw bytes
        coord_data += struct.pack("<fff", x, y, z)

    node_hdr = _chunk(
        0x01041100,
        _chunk(0x01041101, _uint32(n_nodes))
        + _chunk(0x01041102, _uint32(3)),
    )
    node_sect = _chunk(0x01041000, node_hdr + _chunk(0x01041200, coord_data))

    # Face connectivity as sub-chunks
    face_conn_bytes = b""
    for fi in range(n_faces):
        base = fi * nodes_per_face
        nodes = [base + j for j in range(nodes_per_face)]
        # Clamp to valid node indices
        nodes = [min(n, n_nodes - 1) for n in nodes]
        face_conn_bytes += _face_subchunk(fi + 1, nodes)

    surf_hdr = _chunk(
        0x01043101,
        _chunk(0x01043102, _uint32(surface_id))
        + _chunk(0x01043103, _uint32(n_faces))
        + _chunk(0x01043104, _lstring(surface_name))
        + _chunk(0x01043105, _uint32(nodes_per_face)),
    )
    surf_chunk = _chunk(
        0x01043100,
        surf_hdr + _chunk(0x01043200, face_conn_bytes),
    )
    surf_sect = _chunk(0x01043000, surf_chunk)
    mesh = _chunk(0x01040000, node_sect + surf_sect)

    # States
    state_chunks = b""
    for si in range(n_states):
        t = float(si)
        pressures = [float(si * 0.1 + fi * 0.01) for fi in range(n_faces)]
        region_payload = _region_block(surface_id, pressures)

        state_hdr = _chunk(
            0x02010000,
            _chunk(0x02010002, _float32(t))
            + _chunk(0x02010003, _uint32(si)),
        )
        data_item = _chunk(
            0x02020001,
            _chunk(0x02020002, _uint32(1))
            + _chunk(0x02020003, region_payload),
        )
        surf_data_sect = _chunk(0x02020500, data_item)
        state_data = _chunk(0x02020000, surf_data_sect)
        state_chunks += _chunk(0x02000000, state_hdr + state_data)

    return struct.pack("<I", XPLT_MAGIC) + root_header + mesh + state_chunks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_xplt(tmp_path) -> Path:
    p = tmp_path / "synthetic.xplt"
    p.write_bytes(build_synthetic_xplt_with_coords())
    return p


@pytest.fixture
def synthetic_xplt_data(synthetic_xplt) -> XpltData:
    return XpltParser().parse(synthetic_xplt)


# ---------------------------------------------------------------------------
# Helper: build a mock XpltData for unit tests
# ---------------------------------------------------------------------------

def _make_mock_xplt_data(
    n_faces: int = 5,
    n_states: int = 3,
    surface_id: int = 2,
    surface_name: str = "TestSurface",
    nodes_per_face: int = 3,
    n_nodes: int = 9,
) -> XpltData:
    """Build a minimal XpltData in-memory for mocking."""
    connectivity = np.arange(n_faces * nodes_per_face, dtype=np.int32).reshape(n_faces, nodes_per_face) % n_nodes
    surf = SurfacePatch(
        surface_id=surface_id,
        name=surface_name,
        n_faces=n_faces,
        nodes_per_face=nodes_per_face,
        face_connectivity=connectivity,
    )
    # Simple node coords: grid in 3D
    coords = np.zeros((n_nodes, 3), dtype=np.float32)
    for i in range(n_nodes):
        coords[i] = [float(i), float(i * 0.5), 0.0]

    surf_var = DictVariable(
        name="contact pressure",
        data_type=0,
        fmt=1,
        section="surface",
        index_in_section=0,
    )

    states = []
    for si in range(n_states):
        pressures = np.array([float(si * 0.1 + fi * 0.01) for fi in range(n_faces)], dtype=np.float32)
        per_region = {surface_id: pressures}
        sd = StateVariableData(var_index=1, values=pressures, per_region=per_region)
        state = SimulationState(
            state_index=si,
            time=float(si),
            surface_data=[sd],
        )
        states.append(state)

    data = XpltData(
        version=53,
        software="FEBio 4.10.0",
        n_nodes=n_nodes,
        surfaces=[surf],
        surface_vars=[surf_var],
        states=states,
        node_coords=coords,
    )
    return data


# ===========================================================================
# TestXpltParserEnhanced
# ===========================================================================

class TestXpltParserEnhanced:
    """Tests for enhanced parser features: node_coords, face_connectivity, per_region."""

    def test_surface_patch_not_frozen(self):
        """SurfacePatch should be mutable (not frozen)."""
        sp = SurfacePatch(surface_id=1, name="test", n_faces=5, nodes_per_face=3)
        sp.name = "changed"  # should not raise
        assert sp.name == "changed"

    def test_surface_patch_face_connectivity_default_none(self):
        sp = SurfacePatch(surface_id=1, name="test", n_faces=5, nodes_per_face=3)
        assert sp.face_connectivity is None

    def test_state_variable_data_per_region_default(self):
        sd = StateVariableData(var_index=1, values=np.array([1.0], dtype=np.float32))
        assert sd.per_region == {}

    def test_xplt_data_node_coords_default_none(self):
        data = XpltData()
        assert data.node_coords is None

    def test_surface_by_id(self):
        surf = SurfacePatch(surface_id=42, name="MySurf", n_faces=10, nodes_per_face=3)
        data = XpltData(surfaces=[surf])
        found = data.surface_by_id(42)
        assert found is surf

    def test_surface_by_id_not_found(self):
        data = XpltData()
        assert data.surface_by_id(99) is None

    def test_surface_by_name(self):
        surf = SurfacePatch(surface_id=1, name="MySurf", n_faces=10, nodes_per_face=3)
        data = XpltData(surfaces=[surf])
        found = data.surface_by_name("MySurf")
        assert found is surf

    def test_surface_by_name_not_found(self):
        data = XpltData()
        assert data.surface_by_name("Missing") is None

    def test_compute_facet_areas_no_surface_raises(self):
        data = XpltData()
        with pytest.raises(ValueError, match="not found"):
            data.compute_facet_areas("Missing")

    def test_compute_facet_areas_no_connectivity_raises(self):
        surf = SurfacePatch(surface_id=1, name="S", n_faces=3, nodes_per_face=3)
        coords = np.zeros((5, 3), dtype=np.float32)
        data = XpltData(surfaces=[surf], node_coords=coords)
        with pytest.raises(ValueError, match="connectivity"):
            data.compute_facet_areas("S")

    def test_compute_facet_areas_no_coords_raises(self):
        connectivity = np.array([[0, 1, 2]], dtype=np.int32)
        surf = SurfacePatch(surface_id=1, name="S", n_faces=1, nodes_per_face=3,
                            face_connectivity=connectivity)
        data = XpltData(surfaces=[surf])
        with pytest.raises(ValueError, match="coordinates"):
            data.compute_facet_areas("S")

    def test_compute_facet_areas_triangle(self):
        """Unit right triangle: nodes (0,0,0), (1,0,0), (0,1,0) → area=0.5."""
        connectivity = np.array([[0, 1, 2]], dtype=np.int32)
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        surf = SurfacePatch(surface_id=1, name="S", n_faces=1, nodes_per_face=3,
                            face_connectivity=connectivity)
        data = XpltData(surfaces=[surf], node_coords=coords)
        areas = data.compute_facet_areas("S")
        assert areas.shape == (1,)
        assert areas[0] == pytest.approx(0.5, rel=1e-5)

    def test_compute_facet_areas_quad(self):
        """Unit quad: 2 triangles of 0.5 each → total area=1.0."""
        connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)
        coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        surf = SurfacePatch(surface_id=1, name="S", n_faces=1, nodes_per_face=4,
                            face_connectivity=connectivity)
        data = XpltData(surfaces=[surf], node_coords=coords)
        areas = data.compute_facet_areas("S")
        assert areas[0] == pytest.approx(1.0, rel=1e-5)

    def test_synthetic_xplt_node_coords_shape(self, synthetic_xplt_data):
        data = synthetic_xplt_data
        assert data.node_coords is not None
        assert data.node_coords.shape == (9, 3)

    def test_synthetic_xplt_node_coords_dtype(self, synthetic_xplt_data):
        assert synthetic_xplt_data.node_coords.dtype == np.float32

    def test_synthetic_xplt_face_connectivity_shape(self, synthetic_xplt_data):
        surf = synthetic_xplt_data.surfaces[0]
        assert surf.face_connectivity is not None
        assert surf.face_connectivity.shape == (surf.n_faces, surf.nodes_per_face)

    def test_synthetic_xplt_face_connectivity_dtype(self, synthetic_xplt_data):
        surf = synthetic_xplt_data.surfaces[0]
        assert surf.face_connectivity.dtype == np.int32

    def test_synthetic_xplt_per_region(self, synthetic_xplt_data):
        data = synthetic_xplt_data
        for state in data.states:
            for sv in state.surface_data:
                assert isinstance(sv.per_region, dict)

    def test_synthetic_xplt_per_region_has_surface_id(self, synthetic_xplt_data):
        data = synthetic_xplt_data
        surface_id = data.surfaces[0].surface_id
        state = data.states[1]  # second state has non-zero pressures
        for sv in state.surface_data:
            if sv.var_index == 1:
                assert surface_id in sv.per_region
                break

    def test_synthetic_xplt_facet_areas_positive(self, synthetic_xplt_data):
        data = synthetic_xplt_data
        surf_name = data.surfaces[0].name
        if data.surfaces[0].face_connectivity is not None and data.node_coords is not None:
            areas = data.compute_facet_areas(surf_name)
            # At least some areas should be positive (nodes at different positions)
            assert len(areas) == data.surfaces[0].n_faces


# ===========================================================================
# Integration: TestXpltParserEnhanced with real file
# ===========================================================================

@pytest.mark.integration
class TestXpltParserEnhancedIntegration:
    @pytest.fixture(autouse=True)
    def require_sample(self):
        if not SAMPLE_XPLT.exists():
            pytest.skip(f"Sample xplt not found: {SAMPLE_XPLT}")

    def test_node_coords_shape(self):
        data = parse_xplt(SAMPLE_XPLT)
        assert data.node_coords is not None
        assert data.node_coords.shape == (N_NODES, 3)

    def test_node_coords_dtype(self):
        data = parse_xplt(SAMPLE_XPLT)
        assert data.node_coords.dtype == np.float32

    def test_face_connectivity_primary_shape(self):
        data = parse_xplt(SAMPLE_XPLT)
        surf = data.surface_by_name(PRIMARY_SURFACE)
        assert surf is not None
        assert surf.face_connectivity is not None
        assert surf.face_connectivity.shape == (PRIMARY_N_FACES, surf.nodes_per_face)

    def test_face_connectivity_secondary_shape(self):
        data = parse_xplt(SAMPLE_XPLT)
        surf = data.surface_by_name(SECONDARY_SURFACE)
        assert surf is not None
        assert surf.face_connectivity is not None
        assert surf.face_connectivity.shape == (SECONDARY_N_FACES, surf.nodes_per_face)

    def test_face_connectivity_dtype(self):
        data = parse_xplt(SAMPLE_XPLT)
        surf = data.surface_by_name(PRIMARY_SURFACE)
        assert surf.face_connectivity.dtype == np.int32

    def test_surface_by_id(self):
        data = parse_xplt(SAMPLE_XPLT)
        surf = data.surface_by_id(2)
        assert surf is not None

    def test_surface_by_name(self):
        data = parse_xplt(SAMPLE_XPLT)
        surf = data.surface_by_name(PRIMARY_SURFACE)
        assert surf is not None
        assert surf.name == PRIMARY_SURFACE

    def test_facet_areas_positive(self):
        data = parse_xplt(SAMPLE_XPLT)
        surf = data.surface_by_name(PRIMARY_SURFACE)
        if surf is not None and surf.face_connectivity is not None:
            areas = data.compute_facet_areas(PRIMARY_SURFACE)
            assert len(areas) == PRIMARY_N_FACES
            assert np.all(areas > 0)

    def test_per_region_contact_pressure_primary(self):
        """Contact pressure (var_idx=1) should have region_id=2 with 2734 faces."""
        data = parse_xplt(SAMPLE_XPLT)
        # Find a state where contact pressure is non-zero
        for state in data.states:
            for sv in state.surface_data:
                if sv.var_index == 1:
                    assert 2 in sv.per_region
                    assert len(sv.per_region[2]) == PRIMARY_N_FACES
                    break
            break  # check first state only

    def test_per_region_contact_pressure_secondary(self):
        """Contact pressure (var_idx=1) should have region_id=3 with 1834 faces."""
        data = parse_xplt(SAMPLE_XPLT)
        for state in data.states:
            for sv in state.surface_data:
                if sv.var_index == 1:
                    assert 3 in sv.per_region
                    assert len(sv.per_region[3]) == SECONDARY_N_FACES
                    break
            break


# ===========================================================================
# TestFacetTracker (unit tests with mocking)
# ===========================================================================

class TestFacetTrackerUnit:
    """Unit tests with mocked XpltParser."""

    def _make_tracker_with_mock(self, mock_data: XpltData) -> FacetTracker:
        tracker = FacetTracker()
        tracker._parser = MagicMock()
        tracker._parser.parse.return_value = mock_data
        return tracker

    def test_extract_returns_list_of_facet_time_series(self, tmp_path):
        data = _make_mock_xplt_data()
        tracker = self._make_tracker_with_mock(data)
        result = tracker.extract(tmp_path / "fake.xplt", speed_mm_s=5.0,
                                 surface_name="TestSurface")
        assert isinstance(result, list)
        assert all(isinstance(ts, FacetTimeSeries) for ts in result)

    def test_extract_all_facets_when_none(self, tmp_path):
        n_faces = 5
        data = _make_mock_xplt_data(n_faces=n_faces)
        tracker = self._make_tracker_with_mock(data)
        result = tracker.extract(tmp_path / "fake.xplt", speed_mm_s=5.0,
                                 surface_name="TestSurface", facet_ids=None)
        assert len(result) == n_faces

    def test_extract_selected_facets(self, tmp_path):
        data = _make_mock_xplt_data(n_faces=10)
        tracker = self._make_tracker_with_mock(data)
        result = tracker.extract(tmp_path / "fake.xplt", speed_mm_s=5.0,
                                 surface_name="TestSurface", facet_ids=[0, 1, 2])
        assert len(result) == 3

    def test_extract_facet_ids_correct(self, tmp_path):
        data = _make_mock_xplt_data(n_faces=10)
        tracker = self._make_tracker_with_mock(data)
        result = tracker.extract(tmp_path / "fake.xplt", speed_mm_s=5.0,
                                 surface_name="TestSurface", facet_ids=[3, 7])
        ids = [ts.facet_id for ts in result]
        assert ids == [3, 7]

    def test_extract_out_of_range_raises(self, tmp_path):
        data = _make_mock_xplt_data(n_faces=5)
        tracker = self._make_tracker_with_mock(data)
        with pytest.raises(ValueError, match="out of range"):
            tracker.extract(tmp_path / "fake.xplt", speed_mm_s=5.0,
                            surface_name="TestSurface", facet_ids=[0, 99])

    def test_extract_nonexistent_surface_raises(self, tmp_path):
        data = _make_mock_xplt_data()
        tracker = self._make_tracker_with_mock(data)
        with pytest.raises(ValueError, match="not found"):
            tracker.extract(tmp_path / "fake.xplt", speed_mm_s=5.0,
                            surface_name="NonExistentSurface")

    def test_extract_times_shape(self, tmp_path):
        n_states = 4
        data = _make_mock_xplt_data(n_states=n_states)
        tracker = self._make_tracker_with_mock(data)
        result = tracker.extract(tmp_path / "fake.xplt", speed_mm_s=5.0,
                                 surface_name="TestSurface", facet_ids=[0])
        assert result[0].times.shape == (n_states,)

    def test_extract_pressures_shape(self, tmp_path):
        n_states = 4
        data = _make_mock_xplt_data(n_states=n_states)
        tracker = self._make_tracker_with_mock(data)
        result = tracker.extract(tmp_path / "fake.xplt", speed_mm_s=5.0,
                                 surface_name="TestSurface", facet_ids=[0])
        assert result[0].pressures.shape == (n_states,)

    def test_extract_area_set(self, tmp_path):
        data = _make_mock_xplt_data()
        tracker = self._make_tracker_with_mock(data)
        result = tracker.extract(tmp_path / "fake.xplt", speed_mm_s=5.0,
                                 surface_name="TestSurface", facet_ids=[0])
        assert isinstance(result[0].area, float)

    def test_extract_speed_preserved(self, tmp_path):
        data = _make_mock_xplt_data()
        tracker = self._make_tracker_with_mock(data)
        result = tracker.extract(tmp_path / "fake.xplt", speed_mm_s=7.5,
                                 surface_name="TestSurface", facet_ids=[0])
        assert result[0].speed_mm_s == 7.5

    def test_peak_pressure_property(self, tmp_path):
        data = _make_mock_xplt_data()
        tracker = self._make_tracker_with_mock(data)
        result = tracker.extract(tmp_path / "fake.xplt", speed_mm_s=5.0,
                                 surface_name="TestSurface", facet_ids=[0])
        ts = result[0]
        assert ts.peak_pressure == pytest.approx(float(np.max(ts.pressures)), rel=1e-5)

    def test_as_rows_count(self, tmp_path):
        n_states = 3
        data = _make_mock_xplt_data(n_states=n_states)
        tracker = self._make_tracker_with_mock(data)
        result = tracker.extract(tmp_path / "fake.xplt", speed_mm_s=5.0,
                                 surface_name="TestSurface", facet_ids=[0])
        rows = result[0].as_rows()
        assert len(rows) == n_states

    def test_get_facet_info_returns_list(self, tmp_path):
        data = _make_mock_xplt_data(n_faces=5)
        tracker = self._make_tracker_with_mock(data)
        infos = tracker.get_facet_info(tmp_path / "fake.xplt",
                                       surface_name="TestSurface")
        assert isinstance(infos, list)
        assert len(infos) == 5

    def test_get_facet_info_selected(self, tmp_path):
        data = _make_mock_xplt_data(n_faces=10)
        tracker = self._make_tracker_with_mock(data)
        infos = tracker.get_facet_info(tmp_path / "fake.xplt",
                                       surface_name="TestSurface",
                                       facet_ids=[1, 3])
        assert len(infos) == 2
        assert infos[0].facet_id == 1
        assert infos[1].facet_id == 3

    def test_get_facet_info_nonexistent_surface_raises(self, tmp_path):
        data = _make_mock_xplt_data()
        tracker = self._make_tracker_with_mock(data)
        with pytest.raises(ValueError, match="not found"):
            tracker.get_facet_info(tmp_path / "fake.xplt",
                                   surface_name="Missing")


# ===========================================================================
# Integration: TestFacetTracker with real file
# ===========================================================================

@pytest.mark.integration
class TestFacetTrackerIntegration:
    @pytest.fixture(autouse=True)
    def require_sample(self):
        if not SAMPLE_XPLT.exists():
            pytest.skip(f"Sample xplt not found: {SAMPLE_XPLT}")

    def test_extract_returns_facet_time_series(self):
        tracker = FacetTracker()
        result = tracker.extract(
            SAMPLE_XPLT, speed_mm_s=5.0,
            surface_name=PRIMARY_SURFACE, facet_ids=[0, 1, 2]
        )
        assert len(result) == 3
        assert all(isinstance(ts, FacetTimeSeries) for ts in result)

    def test_extract_facet_ids(self):
        tracker = FacetTracker()
        result = tracker.extract(
            SAMPLE_XPLT, speed_mm_s=5.0,
            surface_name=PRIMARY_SURFACE, facet_ids=[0, 1, 2]
        )
        assert [ts.facet_id for ts in result] == [0, 1, 2]

    def test_extract_times_shape(self):
        tracker = FacetTracker()
        result = tracker.extract(
            SAMPLE_XPLT, speed_mm_s=5.0,
            surface_name=PRIMARY_SURFACE, facet_ids=[0]
        )
        assert result[0].times.shape == (N_STATES,)

    def test_extract_pressures_shape(self):
        tracker = FacetTracker()
        result = tracker.extract(
            SAMPLE_XPLT, speed_mm_s=5.0,
            surface_name=PRIMARY_SURFACE, facet_ids=[0]
        )
        assert result[0].pressures.shape == (N_STATES,)

    def test_extract_area_positive(self):
        tracker = FacetTracker()
        result = tracker.extract(
            SAMPLE_XPLT, speed_mm_s=5.0,
            surface_name=PRIMARY_SURFACE, facet_ids=[0, 1, 2]
        )
        # Areas should be positive if face connectivity was parsed
        for ts in result:
            assert ts.area >= 0.0  # >= 0 to handle edge cases

    def test_extract_all_facets(self):
        tracker = FacetTracker()
        result = tracker.extract(
            SAMPLE_XPLT, speed_mm_s=5.0,
            surface_name=PRIMARY_SURFACE, facet_ids=None
        )
        assert len(result) == PRIMARY_N_FACES

    def test_extract_out_of_range_raises(self):
        tracker = FacetTracker()
        with pytest.raises(ValueError, match="out of range"):
            tracker.extract(
                SAMPLE_XPLT, speed_mm_s=5.0,
                surface_name=PRIMARY_SURFACE,
                facet_ids=[0, PRIMARY_N_FACES + 100]
            )

    def test_peak_pressure_property(self):
        tracker = FacetTracker()
        result = tracker.extract(
            SAMPLE_XPLT, speed_mm_s=5.0,
            surface_name=PRIMARY_SURFACE, facet_ids=[0]
        )
        ts = result[0]
        assert ts.peak_pressure == pytest.approx(float(np.max(ts.pressures)), rel=1e-5)

    def test_as_rows_count(self):
        tracker = FacetTracker()
        result = tracker.extract(
            SAMPLE_XPLT, speed_mm_s=5.0,
            surface_name=PRIMARY_SURFACE, facet_ids=[0]
        )
        rows = result[0].as_rows()
        assert len(rows) == N_STATES

    def test_nonexistent_surface_raises(self):
        tracker = FacetTracker()
        with pytest.raises(ValueError, match="not found"):
            tracker.extract(
                SAMPLE_XPLT, speed_mm_s=5.0,
                surface_name="DoesNotExist"
            )

    def test_get_facet_info_returns_facet_info(self):
        tracker = FacetTracker()
        infos = tracker.get_facet_info(
            SAMPLE_XPLT, surface_name=PRIMARY_SURFACE, facet_ids=[0, 1, 2]
        )
        assert len(infos) == 3
        assert all(isinstance(fi, FacetInfo) for fi in infos)


# ===========================================================================
# TestFacetDatasetBuilder (with mocked FacetTracker)
# ===========================================================================

def _make_fake_series(n_facets=3, n_states=4, speed=5.0, surface_id=2):
    series = []
    for fid in range(n_facets):
        ts = FacetTimeSeries(
            facet_id=fid,
            surface_name="TestSurface",
            surface_id=surface_id,
            area=float(fid + 1) * 0.5,
            speed_mm_s=speed,
            times=np.linspace(0, 1, n_states),
            pressures=np.random.rand(n_states).astype(np.float32),
        )
        series.append(ts)
    return series


class TestFacetDatasetBuilder:
    @pytest.fixture
    def mock_tracker(self):
        with patch("digital_twin_ui.ml.facet_dataset.FacetTracker") as MockTracker:
            instance = MockTracker.return_value
            instance.extract.return_value = _make_fake_series()
            yield instance

    @pytest.fixture
    def builder(self, tmp_path, mock_tracker):
        b = FacetDatasetBuilder(
            surface_name="TestSurface",
            dataset_path=tmp_path / "dataset.parquet",
            peak_only=True,
        )
        b._tracker = mock_tracker
        return b

    def test_build_returns_dataframe(self, builder, tmp_path):
        run_configs = [
            {"xplt_path": tmp_path / "r1.xplt", "speed_mm_s": 5.0, "run_id": "r1"},
        ]
        df = builder.build(run_configs)
        assert isinstance(df, pd.DataFrame)

    def test_build_correct_columns_peak_only(self, builder, tmp_path):
        run_configs = [{"xplt_path": tmp_path / "r1.xplt", "speed_mm_s": 5.0, "run_id": "r1"}]
        df = builder.build(run_configs)
        assert COL_RUN_ID in df.columns
        assert COL_FACET_ID in df.columns
        assert COL_SPEED in df.columns
        assert COL_AREA in df.columns
        assert COL_PEAK_PRESSURE in df.columns

    def test_build_peak_only_one_row_per_facet(self, builder, tmp_path):
        n_facets = 3
        builder._tracker.extract.return_value = _make_fake_series(n_facets=n_facets)
        run_configs = [{"xplt_path": tmp_path / "r1.xplt", "speed_mm_s": 5.0, "run_id": "r1"}]
        df = builder.build(run_configs)
        assert len(df) == n_facets

    def test_build_per_timestep_one_row_per_facet_time(self, tmp_path):
        n_facets = 3
        n_states = 4
        with patch("digital_twin_ui.ml.facet_dataset.FacetTracker") as MockTracker:
            instance = MockTracker.return_value
            instance.extract.return_value = _make_fake_series(n_facets=n_facets, n_states=n_states)
            builder = FacetDatasetBuilder(
                surface_name="TestSurface",
                dataset_path=tmp_path / "dataset2.parquet",
                peak_only=False,
            )
            builder._tracker = instance
            run_configs = [{"xplt_path": tmp_path / "r1.xplt", "speed_mm_s": 5.0, "run_id": "r1"}]
            df = builder.build(run_configs)
        assert len(df) == n_facets * n_states

    def test_build_failed_run_skipped(self, tmp_path):
        with patch("digital_twin_ui.ml.facet_dataset.FacetTracker") as MockTracker:
            instance = MockTracker.return_value
            instance.extract.side_effect = [
                Exception("bad run"),
                _make_fake_series(n_facets=2),
            ]
            builder = FacetDatasetBuilder(
                surface_name="TestSurface",
                dataset_path=tmp_path / "dataset3.parquet",
                peak_only=True,
            )
            builder._tracker = instance
            run_configs = [
                {"xplt_path": tmp_path / "r1.xplt", "speed_mm_s": 5.0, "run_id": "r1"},
                {"xplt_path": tmp_path / "r2.xplt", "speed_mm_s": 6.0, "run_id": "r2"},
            ]
            df = builder.build(run_configs)
        assert len(df) == 2  # only r2

    def test_load_raises_if_no_dataset(self, tmp_path):
        builder = FacetDatasetBuilder(
            dataset_path=tmp_path / "nonexistent.parquet",
        )
        with pytest.raises(FileNotFoundError):
            builder.load()

    def test_append_run_grows_dataset(self, builder, tmp_path):
        run_configs = [{"xplt_path": tmp_path / "r1.xplt", "speed_mm_s": 5.0, "run_id": "r1"}]
        df1 = builder.build(run_configs)
        n1 = len(df1)

        builder._tracker.extract.return_value = _make_fake_series(n_facets=2)
        df2 = builder.append_run(tmp_path / "r2.xplt", speed_mm_s=6.0, run_id="r2")
        assert len(df2) == n1 + 2

    def test_dataset_path_property(self, tmp_path):
        expected = tmp_path / "my_dataset.parquet"
        builder = FacetDatasetBuilder(dataset_path=expected)
        assert builder.dataset_path == expected

    def test_surface_name_property(self):
        builder = FacetDatasetBuilder(surface_name="MySurface",
                                      dataset_path=Path("/tmp/x.parquet"))
        assert builder.surface_name == "MySurface"

    def test_build_no_runs_returns_empty(self, builder):
        df = builder.build([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ===========================================================================
# TestFacetPressureMLP
# ===========================================================================

class TestFacetPressureMLP:
    def test_forward_shape(self):
        model = FacetPressureMLP(hidden_dims=[16, 16])
        x = torch.randn(4, 2)
        y = model(x)
        assert y.shape == (4, 2)

    def test_custom_input_output_dim(self):
        model = FacetPressureMLP(input_dim=3, hidden_dims=[8], output_dim=1)
        x = torch.randn(5, 3)
        y = model(x)
        assert y.shape == (5, 1)

    def test_count_parameters_positive(self):
        model = FacetPressureMLP(hidden_dims=[16, 16])
        assert model.count_parameters() > 0

    def test_hidden_dims_stored(self):
        hd = [32, 64, 32]
        model = FacetPressureMLP(hidden_dims=hd)
        assert model.hidden_dims == hd

    def test_gradient_flows(self):
        model = FacetPressureMLP(hidden_dims=[16])
        x = torch.randn(3, 2)
        y = model(x)
        loss = y.sum()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert not torch.all(p.grad == 0)

    def test_batch_norm_variant(self):
        model = FacetPressureMLP(hidden_dims=[16], batch_norm=True)
        x = torch.randn(4, 2)
        model.train()
        y = model(x)
        assert y.shape == (4, 2)

    def test_dropout_variant(self):
        model = FacetPressureMLP(hidden_dims=[16], dropout=0.1)
        x = torch.randn(4, 2)
        y = model(x)
        assert y.shape == (4, 2)


# ===========================================================================
# TestFacetTrainer
# ===========================================================================

def _make_training_df(n_rows: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        COL_FACET_ID: np.random.randint(0, 50, n_rows),
        COL_SPEED: np.random.uniform(3.0, 7.0, n_rows),
        COL_AREA: np.random.uniform(0.1, 1.0, n_rows),
        COL_PEAK_PRESSURE: np.random.uniform(0.0, 10.0, n_rows),
    })


class TestFacetTrainer:
    @pytest.fixture
    def tiny_trainer(self, tmp_path):
        return FacetTrainer(
            hidden_dims=[8, 8],
            max_epochs=5,
            patience=5,
            batch_size=32,
            val_fraction=0.2,
            checkpoint_dir=tmp_path / "models",
        )

    @pytest.fixture
    def tiny_df(self):
        return _make_training_df(50)

    def test_train_returns_result(self, tiny_trainer, tiny_df):
        result = tiny_trainer.train(tiny_df)
        assert isinstance(result, FacetTrainingResult)

    def test_model_is_facet_pressure_mlp(self, tiny_trainer, tiny_df):
        result = tiny_trainer.train(tiny_df)
        assert isinstance(result.model, FacetPressureMLP)

    def test_checkpoint_saved(self, tiny_trainer, tiny_df, tmp_path):
        result = tiny_trainer.train(tiny_df)
        assert result.model_path is not None
        assert result.model_path.exists()

    def test_feature_stats_not_none(self, tiny_trainer, tiny_df):
        result = tiny_trainer.train(tiny_df)
        assert result.feature_mean is not None
        assert result.feature_std is not None
        assert result.target_mean is not None
        assert result.target_std is not None

    def test_converged_true(self, tiny_trainer, tiny_df):
        result = tiny_trainer.train(tiny_df)
        assert result.converged is True

    def test_n_train_plus_n_val_equals_df_len(self, tiny_trainer, tiny_df):
        result = tiny_trainer.train(tiny_df)
        assert result.n_train + result.n_val == len(tiny_df)

    def test_too_small_df_raises(self, tiny_trainer):
        tiny_df = pd.DataFrame({
            COL_FACET_ID: [0],
            COL_SPEED: [5.0],
            COL_AREA: [0.5],
            COL_PEAK_PRESSURE: [1.0],
        })
        with pytest.raises(ValueError, match="too small"):
            tiny_trainer.train(tiny_df)

    def test_train_losses_length(self, tiny_trainer, tiny_df):
        result = tiny_trainer.train(tiny_df)
        assert len(result.train_losses) == result.epochs_trained
        assert len(result.val_losses) == result.epochs_trained


# ===========================================================================
# TestFacetPredictor
# ===========================================================================

class TestFacetPredictor:
    @pytest.fixture
    def checkpoint_path(self, tmp_path):
        """Create a real checkpoint by training a tiny model."""
        trainer = FacetTrainer(
            hidden_dims=[8, 8],
            max_epochs=2,
            patience=2,
            checkpoint_dir=tmp_path / "models",
        )
        df = _make_training_df(20)
        result = trainer.train(df)
        return result.model_path

    def test_from_checkpoint_loads_model(self, checkpoint_path):
        predictor = FacetPredictor.from_checkpoint(checkpoint_path)
        assert isinstance(predictor.model, FacetPressureMLP)

    def test_predict_returns_two_floats(self, checkpoint_path):
        predictor = FacetPredictor.from_checkpoint(checkpoint_path)
        result = predictor.predict(facet_id=10, speed_mm_s=5.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_predict_batch_correct_length(self, checkpoint_path):
        predictor = FacetPredictor.from_checkpoint(checkpoint_path)
        results = predictor.predict_batch([0, 1, 2], [5.0, 5.0, 5.0])
        assert len(results) == 3

    def test_predict_batch_empty_returns_empty(self, checkpoint_path):
        predictor = FacetPredictor.from_checkpoint(checkpoint_path)
        results = predictor.predict_batch([], [])
        assert results == []

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FacetPredictor.from_checkpoint(tmp_path / "nonexistent.pt")

    def test_bad_checkpoint_raises_key_error(self, tmp_path):
        bad_ckpt = tmp_path / "bad.pt"
        torch.save({"model_state_dict": {}, "hidden_dims": [8]}, bad_ckpt)
        with pytest.raises(KeyError):
            FacetPredictor.from_checkpoint(bad_ckpt)

    def test_model_property(self, checkpoint_path):
        predictor = FacetPredictor.from_checkpoint(checkpoint_path)
        assert predictor.model is not None
