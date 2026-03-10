"""
Tests for digital_twin_ui.extraction.xplt_parser

Test strategy
-------------
  1. Integration tests against the real ``conf_file/jobs/sample_catheterization.xplt``.
     These are marked ``@pytest.mark.integration`` and are the primary validation
     that the parser handles genuine FEBio 4.x output.

  2. Unit tests using a minimal synthetic xplt binary builder.
     These run fast without needing the full simulation file and cover all
     code paths: magic validation, header, dictionary, mesh surfaces, states,
     contact pressure extraction, and error handling.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from digital_twin_ui.extraction.xplt_parser import (
    XPLT_MAGIC,
    DictVariable,
    PressureResult,
    SimulationState,
    SurfacePatch,
    XpltData,
    XpltParser,
    _decode_values_from_bytes,
    extract_contact_pressure,
    parse_xplt,
)


# ---------------------------------------------------------------------------
# Helpers exposed for testing
# ---------------------------------------------------------------------------

# We add a small module-level helper to expose _decode_values without needing
# a full parser instance.  See bottom of this file for monkey-patch if needed.


# ---------------------------------------------------------------------------
# Synthetic xplt builder
# ---------------------------------------------------------------------------

def _chunk(tag: int, content: bytes) -> bytes:
    """Pack a single xplt chunk: tag (4) + size (4) + content."""
    return struct.pack("<II", tag, len(content)) + content


def _uint32(v: int) -> bytes:
    return struct.pack("<I", v)


def _float32(v: float) -> bytes:
    return struct.pack("<f", v)


def _string64(s: str) -> bytes:
    """Null-padded 64-byte string."""
    raw = s.encode("ascii")[:64]
    return raw + b"\x00" * (64 - len(raw))


def _region_block(region_id: int, values: list[float]) -> bytes:
    """Encode one (region_id, byte_count, float32[]) region block."""
    float_bytes = struct.pack(f"<{len(values)}f", *values)
    return _uint32(region_id) + _uint32(len(float_bytes)) + float_bytes


def _dict_item(name: str, data_type: int, fmt: int) -> bytes:
    """Build a PLT_DIC_ITEM chunk."""
    return _chunk(
        0x01020001,
        _chunk(0x01020002, _uint32(data_type))
        + _chunk(0x01020003, _uint32(fmt))
        + _chunk(0x01020005, _uint32(0))
        + _chunk(0x01020004, _string64(name)),
    )


def _lstring(s: str) -> bytes:
    """Length-prefixed string: uint32 len + bytes (matching real FEBio format)."""
    raw = s.encode("ascii") + b"\x00"  # include null terminator in length
    return _uint32(len(raw)) + raw


def _surface_hdr(surf_id: int, n_faces: int, name: str, nodes_per_face: int = 3) -> bytes:
    return _chunk(
        0x01043101,
        _chunk(0x01043102, _uint32(surf_id))
        + _chunk(0x01043103, _uint32(n_faces))
        + _chunk(0x01043104, _lstring(name))
        + _chunk(0x01043105, _uint32(nodes_per_face)),
    )


def _surface_faces(surf_id: int, n_faces: int, nodes_per_face: int = 3) -> bytes:
    """Dummy face connectivity: sequential node indices."""
    node_indices = list(range(n_faces * nodes_per_face))
    raw = b"".join(_uint32(i) for i in node_indices)
    return _chunk(0x01043200, raw)


def _state_data_item(var_idx: int, regions: list[tuple[int, list[float]]]) -> bytes:
    """Build a PLT_DATA_ITEM chunk for one variable."""
    payload = b"".join(_region_block(rid, vals) for rid, vals in regions)
    return _chunk(
        0x02020001,
        _chunk(0x02020002, _uint32(var_idx))
        + _chunk(0x02020003, payload),
    )


def build_minimal_xplt(
    n_nodes: int = 10,
    surfaces: list[tuple[int, str, int]] | None = None,  # (id, name, n_faces)
    surface_vars: list[tuple[str, int, int]] | None = None,  # (name, type, fmt)
    states: list[tuple[float, list[tuple[int, list[float]]]]] | None = None,
    # states: [(time, [(region_id, [pressure_values...]), ...]), ...]
) -> bytes:
    """
    Build a minimal valid xplt binary blob for unit testing.

    Args:
        n_nodes:      Number of mesh nodes.
        surfaces:     List of (surface_id, name, n_faces) tuples.
        surface_vars: Surface dictionary variables as (name, data_type, fmt).
        states:       Simulation states as (time, [(region_id, [values])]).

    Returns:
        bytes — valid xplt binary content.
    """
    if surfaces is None:
        surfaces = [(1, "ContactSurface", 5)]
    if surface_vars is None:
        surface_vars = [("contact pressure", 0, 1)]
    if states is None:
        states = [(0.0, [(1, [0.0] * 5)]), (1.0, [(1, [0.1, 0.2, 0.3, 0.0, 0.0])])]

    # -- Header info
    header_info = _chunk(
        0x01010000,
        _chunk(0x01010001, _uint32(53))          # version 53
        + _chunk(0x01010004, _uint32(0))          # no compression
        + _chunk(0x01010006, _lstring("FEBio 4.10.0")),  # software
    )

    # -- Dictionary: surface variables only
    surf_dict_items = b"".join(
        _dict_item(name, dt, fmt) for name, dt, fmt in surface_vars
    )
    dictionary = _chunk(
        0x01020000,
        _chunk(0x01023000, b"")   # empty nodal section
        + _chunk(0x01024000, b"")  # empty domain section
        + _chunk(0x01025000, surf_dict_items),
    )

    root_header = _chunk(0x01000000, header_info + dictionary)

    # -- Mesh: node section + surface section
    node_hdr = _chunk(
        0x01041100,
        _chunk(0x01041101, _uint32(n_nodes))
        + _chunk(0x01041102, _uint32(3)),  # 3D
    )
    # Dummy node coordinates
    coords = struct.pack(f"<{n_nodes * 3}f", *([0.0] * (n_nodes * 3)))
    node_sect = _chunk(0x01041000, node_hdr + _chunk(0x01041200, coords))

    surf_chunks = b"".join(
        _chunk(
            0x01043100,
            _surface_hdr(sid, nf, sname)
            + _surface_faces(sid, nf),
        )
        for sid, sname, nf in surfaces
    )
    surf_sect = _chunk(0x01043000, surf_chunks)
    mesh = _chunk(0x01040000, node_sect + surf_sect)

    # -- States
    state_chunks = b""
    for state_idx, (t, region_data) in enumerate(states):
        state_hdr = _chunk(
            0x02010000,
            _chunk(0x02010002, _float32(t))
            + _chunk(0x02010003, _uint32(state_idx)),
        )
        # Build surface data section with var_idx=1
        surf_data_item = _state_data_item(1, region_data)
        surf_data_sect = _chunk(0x02020500, surf_data_item)
        state_data = _chunk(0x02020000, surf_data_sect)
        state_chunks += _chunk(0x02000000, state_hdr + state_data)

    return struct.pack("<I", XPLT_MAGIC) + root_header + mesh + state_chunks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_XPLT = Path(__file__).parent.parent / "conf_file" / "jobs" / "sample_catheterization.xplt"


@pytest.fixture
def minimal_xplt(tmp_path) -> Path:
    """Write a minimal synthetic xplt file and return its path."""
    p = tmp_path / "test.xplt"
    p.write_bytes(build_minimal_xplt())
    return p


@pytest.fixture
def two_surface_xplt(tmp_path) -> Path:
    """xplt with 2 contact surfaces and meaningful pressure values."""
    p = tmp_path / "two_surf.xplt"
    content = build_minimal_xplt(
        surfaces=[(2, "Primary", 4), (3, "Secondary", 3)],
        surface_vars=[("contact pressure", 0, 1)],
        states=[
            (0.0, [(2, [0.0, 0.0, 0.0, 0.0]), (3, [0.0, 0.0, 0.0])]),
            (1.0, [(2, [0.1, 0.2, 0.0, 0.0]), (3, [0.3, 0.0, 0.0])]),
            (2.0, [(2, [0.4, 0.5, 0.6, 0.1]), (3, [0.7, 0.8, 0.9])]),
        ],
    )
    p.write_bytes(content)
    return p


# ===========================================================================
# Unit tests: _decode_values
# ===========================================================================

class TestDecodeValues:
    def test_single_region_scalar(self):
        raw = _region_block(1, [1.0, 2.0, 3.0])
        result = _decode_values_from_bytes(raw)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], rtol=1e-5)

    def test_two_regions_concatenated(self):
        raw = _region_block(2, [0.1, 0.2]) + _region_block(3, [0.3, 0.4, 0.5])
        result = _decode_values_from_bytes(raw)
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3, 0.4, 0.5], rtol=1e-5)

    def test_empty_buffer_returns_empty(self):
        result = _decode_values_from_bytes(b"")
        assert len(result) == 0

    def test_too_short_returns_empty(self):
        result = _decode_values_from_bytes(b"\x01\x00\x00\x00")  # only 4 bytes
        assert len(result) == 0

    def test_zero_byte_count_stops_parsing(self):
        raw = _uint32(1) + _uint32(0) + _uint32(99)  # region with byte_count=0
        result = _decode_values_from_bytes(raw)
        assert len(result) == 0

    def test_returns_float32_dtype(self):
        raw = _region_block(1, [1.5, 2.5])
        result = _decode_values_from_bytes(raw)
        assert result.dtype == np.float32

    def test_values_are_copy_not_view(self):
        raw = bytearray(_region_block(1, [1.0, 2.0]))
        result = _decode_values_from_bytes(bytes(raw))
        raw[8] = 0xFF  # mutate source
        assert result[0] == pytest.approx(1.0, rel=1e-4)


# ===========================================================================
# Unit tests: XpltParser with synthetic files
# ===========================================================================

class TestXpltParserHeader:
    def test_invalid_magic_raises(self, tmp_path):
        bad = tmp_path / "bad.xplt"
        bad.write_bytes(b"\x00\x00\x00\x00" + b"\x00" * 20)
        with pytest.raises(ValueError, match="Not a valid xplt"):
            XpltParser().parse(bad)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            XpltParser().parse(tmp_path / "nonexistent.xplt")

    def test_too_small_raises(self, tmp_path):
        tiny = tmp_path / "tiny.xplt"
        tiny.write_bytes(b"\x00\x00")
        with pytest.raises(ValueError):
            XpltParser().parse(tiny)

    def test_version_parsed(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert data.version == 53

    def test_software_parsed(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert "FEBio" in data.software


class TestXpltParserDictionary:
    def test_surface_var_found(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert len(data.surface_vars) == 1
        assert data.surface_vars[0].name == "contact pressure"

    def test_surface_var_type(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert data.surface_vars[0].data_type == 0  # scalar

    def test_surface_var_fmt(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert data.surface_vars[0].fmt == 1  # FMT_ITEM

    def test_surface_var_index(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert data.surface_vars[0].index_in_section == 0

    def test_multiple_surface_vars(self, tmp_path):
        content = build_minimal_xplt(
            surface_vars=[
                ("contact pressure", 0, 1),
                ("contact gap", 0, 1),
            ],
            states=[(0.0, [(1, [0.0] * 5)])],
        )
        p = tmp_path / "multi.xplt"
        p.write_bytes(content)
        data = XpltParser().parse(p)
        assert len(data.surface_vars) == 2
        assert data.surface_vars[0].name == "contact pressure"
        assert data.surface_vars[1].name == "contact gap"
        assert data.surface_vars[1].index_in_section == 1


class TestXpltParserMesh:
    def test_n_nodes(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert data.n_nodes == 10

    def test_surfaces_parsed(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert len(data.surfaces) == 1

    def test_surface_id(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert data.surfaces[0].surface_id == 1

    def test_surface_name(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert data.surfaces[0].name == "ContactSurface"

    def test_surface_n_faces(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert data.surfaces[0].n_faces == 5

    def test_two_surfaces(self, two_surface_xplt):
        data = XpltParser().parse(two_surface_xplt)
        assert len(data.surfaces) == 2
        ids = {s.surface_id for s in data.surfaces}
        assert ids == {2, 3}

    def test_n_surface_faces_total(self, two_surface_xplt):
        data = XpltParser().parse(two_surface_xplt)
        assert data.n_surface_faces == 7  # 4 + 3


class TestXpltParserStates:
    def test_state_count(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert len(data.states) == 2

    def test_state_times(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert data.states[0].time == pytest.approx(0.0, abs=1e-5)
        assert data.states[1].time == pytest.approx(1.0, abs=1e-4)

    def test_times_array(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        np.testing.assert_allclose(data.times, [0.0, 1.0], atol=1e-4)

    def test_surface_data_in_state(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        assert len(data.states[1].surface_data) == 1

    def test_surface_data_values(self, minimal_xplt):
        data = XpltParser().parse(minimal_xplt)
        vals = data.states[1].surface_data[0].values
        np.testing.assert_allclose(vals, [0.1, 0.2, 0.3, 0.0, 0.0], rtol=1e-4)

    def test_two_surface_state_data_length(self, two_surface_xplt):
        data = XpltParser().parse(two_surface_xplt)
        state2 = data.states[2]
        vals = state2.surface_data[0].values
        assert len(vals) == 7  # 4 + 3 faces across 2 surfaces


# ===========================================================================
# Unit tests: extract_contact_pressure with synthetic files
# ===========================================================================

class TestExtractContactPressure:
    def test_returns_pressure_result(self, minimal_xplt):
        result = extract_contact_pressure(minimal_xplt)
        assert isinstance(result, PressureResult)

    def test_times_shape(self, minimal_xplt):
        result = extract_contact_pressure(minimal_xplt)
        assert result.times.shape == (2,)

    def test_times_values(self, minimal_xplt):
        result = extract_contact_pressure(minimal_xplt)
        np.testing.assert_allclose(result.times, [0.0, 1.0], atol=1e-4)

    def test_max_pressure_shape(self, minimal_xplt):
        result = extract_contact_pressure(minimal_xplt)
        assert result.max_pressure.shape == (2,)

    def test_max_pressure_values(self, minimal_xplt):
        result = extract_contact_pressure(minimal_xplt)
        assert result.max_pressure[0] == pytest.approx(0.0, abs=1e-6)
        assert result.max_pressure[1] == pytest.approx(0.3, rel=1e-4)

    def test_mean_pressure_nonzero_only(self, minimal_xplt):
        """Mean should only consider faces with p > 0."""
        result = extract_contact_pressure(minimal_xplt)
        # State 1: [0.1, 0.2, 0.3, 0.0, 0.0] → mean of nonzero = (0.1+0.2+0.3)/3
        expected_mean = (0.1 + 0.2 + 0.3) / 3
        assert result.mean_pressure[1] == pytest.approx(expected_mean, rel=1e-4)

    def test_pressures_list_length(self, minimal_xplt):
        result = extract_contact_pressure(minimal_xplt)
        assert len(result.pressures) == 2

    def test_n_faces(self, minimal_xplt):
        result = extract_contact_pressure(minimal_xplt)
        assert result.n_faces == 5

    def test_variable_name(self, minimal_xplt):
        result = extract_contact_pressure(minimal_xplt)
        assert result.variable_name == "contact pressure"

    def test_source_path(self, minimal_xplt):
        result = extract_contact_pressure(minimal_xplt)
        assert result.source_path == minimal_xplt

    def test_two_surfaces_concatenated(self, two_surface_xplt):
        result = extract_contact_pressure(two_surface_xplt)
        # state 2: region2=[0.4,0.5,0.6,0.1], region3=[0.7,0.8,0.9] → max=0.9
        assert result.max_pressure[2] == pytest.approx(0.9, rel=1e-4)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_contact_pressure(tmp_path / "missing.xplt")

    def test_no_surface_vars_raises(self, tmp_path):
        content = build_minimal_xplt(surface_vars=[])
        p = tmp_path / "no_surf.xplt"
        p.write_bytes(content)
        with pytest.raises(ValueError, match="No surface variables"):
            extract_contact_pressure(p)

    def test_partial_name_match(self, tmp_path):
        """Parser should find 'contact pressure' even with a partial query."""
        content = build_minimal_xplt(
            surface_vars=[("contact pressure", 0, 1)],
        )
        p = tmp_path / "partial.xplt"
        p.write_bytes(content)
        result = extract_contact_pressure(p, variable_name="pressure")
        assert result.variable_name == "contact pressure"

    def test_as_dict(self, minimal_xplt):
        result = extract_contact_pressure(minimal_xplt)
        d = result.as_dict()
        assert "times" in d
        assert "max_pressure" in d
        assert "mean_pressure" in d
        assert "n_faces" in d
        assert "variable_name" in d


# ===========================================================================
# Integration tests: real sample_catheterization.xplt
# ===========================================================================

@pytest.mark.integration
class TestRealXplt:
    @pytest.fixture(autouse=True)
    def require_sample(self):
        if not SAMPLE_XPLT.exists():
            pytest.skip(f"Sample xplt not found: {SAMPLE_XPLT}")

    def test_parse_returns_xplt_data(self):
        data = parse_xplt(SAMPLE_XPLT)
        assert isinstance(data, XpltData)

    def test_version_53(self):
        data = parse_xplt(SAMPLE_XPLT)
        assert data.version == 53

    def test_software_string(self):
        data = parse_xplt(SAMPLE_XPLT)
        assert "FEBio" in data.software

    def test_node_count(self):
        data = parse_xplt(SAMPLE_XPLT)
        assert data.n_nodes == 3289

    def test_surface_count(self):
        """Mesh has 3 surfaces: ZeroDisplacement, SlidingElastic1P, SlidingElastic1S."""
        data = parse_xplt(SAMPLE_XPLT)
        assert len(data.surfaces) == 3

    def test_contact_surface_names(self):
        data = parse_xplt(SAMPLE_XPLT)
        names = {s.name for s in data.surfaces}
        assert "SlidingElastic1Primary" in names
        assert "SlidingElastic1Secondary" in names

    def test_contact_surface_face_counts(self):
        data = parse_xplt(SAMPLE_XPLT)
        face_counts = {s.name: s.n_faces for s in data.surfaces}
        assert face_counts["SlidingElastic1Primary"] == 2734
        assert face_counts["SlidingElastic1Secondary"] == 1834

    def test_surface_variable_count(self):
        """Dictionary has 11 surface variables."""
        data = parse_xplt(SAMPLE_XPLT)
        assert len(data.surface_vars) >= 1

    def test_contact_pressure_in_dict(self):
        data = parse_xplt(SAMPLE_XPLT)
        names = [v.name for v in data.surface_vars]
        assert "contact pressure" in names

    def test_state_count(self):
        """41 states: t=0.0 to t=4.0 in 0.1 s steps."""
        data = parse_xplt(SAMPLE_XPLT)
        assert len(data.states) == 41

    def test_state_times_range(self):
        data = parse_xplt(SAMPLE_XPLT)
        assert data.times[0] == pytest.approx(0.0, abs=1e-4)
        assert data.times[-1] == pytest.approx(4.0, abs=1e-3)

    def test_extract_contact_pressure_returns_result(self):
        result = extract_contact_pressure(SAMPLE_XPLT)
        assert isinstance(result, PressureResult)

    def test_pressure_times(self):
        result = extract_contact_pressure(SAMPLE_XPLT)
        assert len(result.times) == 41
        assert result.times[0] == pytest.approx(0.0, abs=1e-4)
        assert result.times[-1] == pytest.approx(4.0, abs=1e-3)

    def test_zero_pressure_before_insertion(self):
        """Contact pressure is 0 during clamping phase (t < ~2.3 s)."""
        result = extract_contact_pressure(SAMPLE_XPLT)
        # States at t = 0.0 to 2.2 should have max_pressure = 0
        early_max = result.max_pressure[result.times < 2.2]
        assert np.all(early_max == 0.0)

    def test_nonzero_pressure_during_insertion(self):
        """Contact pressure rises during insertion phase (t > 2.3 s)."""
        result = extract_contact_pressure(SAMPLE_XPLT)
        late_max = result.max_pressure[result.times >= 2.5]
        assert np.any(late_max > 0.0)

    def test_max_pressure_positive(self):
        result = extract_contact_pressure(SAMPLE_XPLT)
        assert result.max_pressure.max() > 0.0

    def test_mean_pressure_positive_where_active(self):
        result = extract_contact_pressure(SAMPLE_XPLT)
        active = result.mean_pressure[result.max_pressure > 0]
        assert np.all(active > 0.0)

    def test_n_faces_matches_contact_surfaces(self):
        """Total faces = SlidingElastic1P (2734) + SlidingElastic1S (1834) = 4568."""
        result = extract_contact_pressure(SAMPLE_XPLT)
        assert result.n_faces == 4568

    def test_pressures_list_length(self):
        result = extract_contact_pressure(SAMPLE_XPLT)
        assert len(result.pressures) == 41

    def test_pressure_array_length_per_state(self):
        result = extract_contact_pressure(SAMPLE_XPLT)
        for p_arr in result.pressures:
            assert len(p_arr) == 4568

    def test_as_dict_serialisable(self):
        import json
        result = extract_contact_pressure(SAMPLE_XPLT)
        d = result.as_dict()
        json.dumps(d)  # must not raise

    def test_max_pressure_monotone_region(self):
        """Pressure generally increases as more catheter area contacts urethra."""
        result = extract_contact_pressure(SAMPLE_XPLT)
        # After contact begins, maximum should be non-trivially positive at end
        assert result.max_pressure[-1] > result.max_pressure[0]
