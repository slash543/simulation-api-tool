"""
Tests for scripts/extract_pressure.py

Covers:
  - compute_centroids       : face centroid geometry
  - project_cylindrical     : PCA-based cylindrical unrolling
  - save_csv                : CSV file creation and content
  - plot_static             : static contour PNG creation
  - plot_animation          : animated GIF creation (skipped when Pillow absent)
  - list_surfaces           : surface inspection output (mocked parser)
  - main() CLI              : argument parsing, error paths, output file creation
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load the script module (it lives in scripts/, not a package)
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "extract_pressure.py"
_PROJECT_ROOT = Path(__file__).parent.parent

_spec = importlib.util.spec_from_file_location("extract_pressure", _SCRIPT_PATH)
_ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ep)

# Convenience aliases for the public functions under test
compute_centroids    = _ep.compute_centroids
project_cylindrical  = _ep.project_cylindrical
save_csv             = _ep.save_csv
plot_static          = _ep.plot_static
plot_animation       = _ep.plot_animation
list_surfaces        = _ep.list_surfaces
main                 = _ep.main

# ---------------------------------------------------------------------------
# Constants for integration tests
# ---------------------------------------------------------------------------

SAMPLE_XPLT    = _PROJECT_ROOT / "conf_file" / "jobs" / "sample_catheterization.xplt"
SAMPLE_SURFACE = "SlidingElastic1Primary"
SAMPLE_N_FACES = 2734
SAMPLE_N_STATES = 41


# ---------------------------------------------------------------------------
# Shared synthetic helpers
# ---------------------------------------------------------------------------

def _make_face_connectivity(n_faces: int, n_nodes: int, nodes_per_face: int = 3) -> np.ndarray:
    """Simple cycling connectivity — every entry is in [0, n_nodes)."""
    idx = np.arange(n_faces * nodes_per_face, dtype=np.int32) % n_nodes
    return idx.reshape(n_faces, nodes_per_face)


def _make_node_coords(n_nodes: int) -> np.ndarray:
    """Grid of nodes in a plane (for deterministic centroid tests)."""
    coords = np.zeros((n_nodes, 3), dtype=np.float32)
    for i in range(n_nodes):
        coords[i] = [float(i), float(i * 0.5), 0.0]
    return coords


def _make_cylindrical_centroids(n: int = 60) -> np.ndarray:
    """
    Points on the surface of a cylinder aligned with the Z axis.
    Main PCA axis should therefore be Z.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = np.linspace(0, 10, n)
    return np.column_stack([np.cos(t), np.sin(t), z]).astype(np.float64)


def _make_fake_series(n_facets: int = 4, n_states: int = 3, speed: float = 5.0):
    """Return a list of FacetTimeSeries with synthetic data."""
    from digital_twin_ui.extraction.facet_tracker import FacetTimeSeries
    series = []
    rng = np.random.default_rng(0)
    for fid in range(n_facets):
        series.append(FacetTimeSeries(
            facet_id=fid,
            surface_name="TestSurface",
            surface_id=2,
            area=float(fid + 1) * 0.25,
            speed_mm_s=speed,
            times=np.linspace(0.0, 2.0, n_states),
            pressures=rng.random(n_states).astype(np.float32),
        ))
    return series


def _make_mock_xplt_data(n_faces: int = 4, n_nodes: int = 9, n_states: int = 3,
                         surface_name: str = "TestSurface"):
    """Build a minimal in-memory XpltData for mocking XpltParser.parse()."""
    from digital_twin_ui.extraction.xplt_parser import (
        XpltData, SurfacePatch, DictVariable, SimulationState, StateVariableData,
    )
    connectivity = _make_face_connectivity(n_faces, n_nodes)
    coords = _make_node_coords(n_nodes)
    surf = SurfacePatch(
        surface_id=2, name=surface_name, n_faces=n_faces,
        nodes_per_face=3, face_connectivity=connectivity,
    )
    surf_var = DictVariable(
        name="contact pressure", data_type=0, fmt=1,
        section="surface", index_in_section=0,
    )
    states = []
    for si in range(n_states):
        pressures = np.full(n_faces, float(si) * 0.1, dtype=np.float32)
        per_region = {2: pressures}
        sd = StateVariableData(var_index=1, values=pressures, per_region=per_region)
        states.append(SimulationState(state_index=si, time=float(si), surface_data=[sd]))
    return XpltData(
        version=53, software="FEBio 4.10.0", n_nodes=n_nodes,
        surfaces=[surf], surface_vars=[surf_var],
        node_coords=coords, states=states,
    )


# ===========================================================================
# TestComputeCentroids
# ===========================================================================

class TestComputeCentroids:
    """Unit tests for compute_centroids()."""

    def test_output_shape(self):
        n_faces, n_nodes, npf = 5, 9, 3
        conn = _make_face_connectivity(n_faces, n_nodes, npf)
        coords = _make_node_coords(n_nodes)
        result = compute_centroids(conn, coords)
        assert result.shape == (n_faces, 3)

    def test_single_triangle_known_centroid(self):
        """Triangle (0,0,0)-(3,0,0)-(0,3,0) → centroid (1,1,0)."""
        conn = np.array([[0, 1, 2]], dtype=np.int32)
        coords = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0]], dtype=np.float32)
        result = compute_centroids(conn, coords)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result[0], [1.0, 1.0, 0.0], atol=1e-6)

    def test_all_zero_coords_gives_zero_centroid(self):
        conn = np.array([[0, 1, 2]], dtype=np.int32)
        coords = np.zeros((3, 3), dtype=np.float32)
        result = compute_centroids(conn, coords)
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 0.0])

    def test_quad_face_centroid(self):
        """Unit square (0,0,0)-(1,0,0)-(1,1,0)-(0,1,0) → centroid (0.5,0.5,0)."""
        conn = np.array([[0, 1, 2, 3]], dtype=np.int32)
        coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        result = compute_centroids(conn, coords)
        np.testing.assert_allclose(result[0], [0.5, 0.5, 0.0], atol=1e-6)


# ===========================================================================
# TestProjectCylindrical
# ===========================================================================

class TestProjectCylindrical:
    """Unit tests for project_cylindrical()."""

    def test_returns_four_values(self):
        centroids = _make_cylindrical_centroids()
        result = project_cylindrical(centroids)
        assert len(result) == 4

    def test_axial_shape(self):
        n = 30
        centroids = _make_cylindrical_centroids(n)
        u, v, _, _ = project_cylindrical(centroids)
        assert u.shape == (n,)

    def test_angle_shape(self):
        n = 30
        centroids = _make_cylindrical_centroids(n)
        u, v, _, _ = project_cylindrical(centroids)
        assert v.shape == (n,)

    def test_angle_range(self):
        """Circumferential angle must lie within [-180, +180] degrees."""
        centroids = _make_cylindrical_centroids(60)
        _, v, _, _ = project_cylindrical(centroids)
        assert float(v.min()) >= -180.0 - 1e-6
        assert float(v.max()) <= 180.0 + 1e-6

    def test_labels_are_strings(self):
        centroids = _make_cylindrical_centroids()
        _, _, xlabel, ylabel = project_cylindrical(centroids)
        assert isinstance(xlabel, str) and len(xlabel) > 0
        assert isinstance(ylabel, str) and len(ylabel) > 0

    def test_flat_surface_does_not_crash(self):
        """Points all in the XY plane — PCA still runs without error."""
        rng = np.random.default_rng(1)
        centroids = rng.random((20, 3))
        centroids[:, 2] = 0.0           # flatten Z
        u, v, _, _ = project_cylindrical(centroids)
        assert u.shape == (20,)
        assert v.shape == (20,)


# ===========================================================================
# TestSaveCsv
# ===========================================================================

class TestSaveCsv:
    """Unit tests for save_csv()."""

    def test_file_created(self, tmp_path):
        series = _make_fake_series(n_facets=2, n_states=3)
        out = tmp_path / "out.csv"
        save_csv(series, out)
        assert out.exists()

    def test_row_count(self, tmp_path):
        n_facets, n_states = 3, 4
        series = _make_fake_series(n_facets=n_facets, n_states=n_states)
        out = tmp_path / "out.csv"
        save_csv(series, out)
        with open(out, newline="") as f:
            rows = list(csv.reader(f))
        # header + data rows
        assert len(rows) == 1 + n_facets * n_states

    def test_expected_columns(self, tmp_path):
        series = _make_fake_series()
        out = tmp_path / "out.csv"
        save_csv(series, out)
        with open(out, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        expected = {"facet_id", "surface_name", "surface_id", "speed_mm_s",
                    "facet_area", "time_step", "time_s", "contact_pressure"}
        assert expected.issubset(set(headers))

    def test_creates_parent_directory(self, tmp_path):
        series = _make_fake_series(n_facets=1, n_states=2)
        nested = tmp_path / "sub" / "dir" / "out.csv"
        save_csv(series, nested)
        assert nested.exists()

    def test_empty_series_does_not_create_file(self, tmp_path, capsys):
        out = tmp_path / "out.csv"
        save_csv([], out)
        assert not out.exists()
        captured = capsys.readouterr()
        assert "no data" in captured.out.lower()


# ===========================================================================
# TestPlotStatic
# ===========================================================================

class TestPlotStatic:
    """Unit tests for plot_static()."""

    def _make_plot_inputs(self, n_facets: int = 10, n_states: int = 6):
        centroids = _make_cylindrical_centroids(n_facets)
        u, v, xlabel, ylabel = project_cylindrical(centroids)
        times = np.linspace(0, 2, n_states)
        rng = np.random.default_rng(42)
        pressures = rng.random((n_states, n_facets)).astype(np.float32)
        return u, v, times, pressures, xlabel, ylabel

    def test_png_created(self, tmp_path):
        u, v, times, pm, xl, yl = self._make_plot_inputs()
        out = tmp_path / "contour.png"
        plot_static(u, v, times, pm, xl, yl, "TestSurface", out)
        assert out.exists()

    def test_png_nonzero_size(self, tmp_path):
        u, v, times, pm, xl, yl = self._make_plot_inputs()
        out = tmp_path / "contour.png"
        plot_static(u, v, times, pm, xl, yl, "TestSurface", out)
        assert out.stat().st_size > 0

    def test_creates_output_directory(self, tmp_path):
        u, v, times, pm, xl, yl = self._make_plot_inputs()
        out = tmp_path / "subdir" / "contour.png"
        plot_static(u, v, times, pm, xl, yl, "TestSurface", out)
        assert out.exists()

    def test_n_panels_clamped_to_n_states(self, tmp_path):
        """Requesting more panels than states should not raise."""
        n_states = 3
        u, v, times, pm, xl, yl = self._make_plot_inputs(n_facets=10, n_states=n_states)
        out = tmp_path / "contour.png"
        plot_static(u, v, times, pm, xl, yl, "TestSurface", out, n_panels=20)
        assert out.exists()

    def test_single_state(self, tmp_path):
        u, v, _, _, xl, yl = self._make_plot_inputs(n_facets=10, n_states=1)
        times = np.array([0.0])
        pm = np.random.rand(1, 10).astype(np.float32)
        out = tmp_path / "contour.png"
        plot_static(u, v, times, pm, xl, yl, "TestSurface", out, n_panels=1)
        assert out.exists()


# ===========================================================================
# TestPlotAnimation
# ===========================================================================

class TestPlotAnimation:
    """Tests for plot_animation() — skipped when Pillow is unavailable."""

    @pytest.fixture(autouse=True)
    def require_pillow(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            pytest.skip("Pillow not installed — animation tests skipped")

    def _make_inputs(self, n_facets: int = 10, n_states: int = 4):
        centroids = _make_cylindrical_centroids(n_facets)
        u, v, xl, yl = project_cylindrical(centroids)
        times = np.linspace(0, 1, n_states)
        pm = np.random.rand(n_states, n_facets).astype(np.float32)
        return u, v, times, pm, xl, yl

    def test_gif_created(self, tmp_path):
        u, v, times, pm, xl, yl = self._make_inputs()
        out = tmp_path / "anim.gif"
        plot_animation(u, v, times, pm, xl, yl, "TestSurface", out, fps=2)
        assert out.exists()

    def test_gif_nonzero_size(self, tmp_path):
        u, v, times, pm, xl, yl = self._make_inputs()
        out = tmp_path / "anim.gif"
        plot_animation(u, v, times, pm, xl, yl, "TestSurface", out, fps=2)
        assert out.stat().st_size > 0


# ===========================================================================
# TestListSurfaces
# ===========================================================================

class TestListSurfaces:
    """Tests for list_surfaces() — mocks XpltParser to avoid real file I/O."""

    def _make_mock_parser(self, surface_names=("SurfA", "SurfB"), n_states=5):
        from digital_twin_ui.extraction.xplt_parser import XpltData, SurfacePatch, DictVariable
        surfs = [
            SurfacePatch(surface_id=i + 1, name=name, n_faces=100, nodes_per_face=3)
            for i, name in enumerate(surface_names)
        ]
        surf_var = DictVariable(name="contact pressure", data_type=0, fmt=1,
                                section="surface", index_in_section=0)
        from digital_twin_ui.extraction.xplt_parser import SimulationState
        states = [SimulationState(state_index=i, time=float(i)) for i in range(n_states)]
        return XpltData(
            version=53, software="FEBio 4.10.0", n_nodes=100,
            surfaces=surfs, surface_vars=[surf_var], states=states,
        )

    def test_prints_surface_names(self, tmp_path, capsys):
        dummy = tmp_path / "dummy.xplt"
        dummy.write_bytes(b"\x00" * 4)
        mock_data = self._make_mock_parser(surface_names=["SlidingPrimary", "SlidingSecondary"])
        with patch.object(_ep, "XpltParser") as MockParser:
            MockParser.return_value.parse.return_value = mock_data
            list_surfaces(dummy)
        captured = capsys.readouterr()
        assert "SlidingPrimary" in captured.out
        assert "SlidingSecondary" in captured.out

    def test_prints_surface_variable_names(self, tmp_path, capsys):
        dummy = tmp_path / "dummy.xplt"
        dummy.write_bytes(b"\x00" * 4)
        mock_data = self._make_mock_parser()
        with patch.object(_ep, "XpltParser") as MockParser:
            MockParser.return_value.parse.return_value = mock_data
            list_surfaces(dummy)
        captured = capsys.readouterr()
        assert "contact pressure" in captured.out

    def test_prints_state_count(self, tmp_path, capsys):
        dummy = tmp_path / "dummy.xplt"
        dummy.write_bytes(b"\x00" * 4)
        mock_data = self._make_mock_parser(n_states=7)
        with patch.object(_ep, "XpltParser") as MockParser:
            MockParser.return_value.parse.return_value = mock_data
            list_surfaces(dummy)
        captured = capsys.readouterr()
        assert "7" in captured.out


# ===========================================================================
# TestMainCli
# ===========================================================================

class TestMainCli:
    """Tests for main() via patched sys.argv — no real .xplt parsing."""

    def _patch_main(self, argv, mock_xplt_data, fake_series):
        """Context manager: patch argv + XpltParser + FacetTracker simultaneously."""
        mock_parser_inst = MagicMock()
        mock_parser_inst.parse.return_value = mock_xplt_data

        mock_tracker_inst = MagicMock()
        mock_tracker_inst.extract.return_value = fake_series

        return (
            patch("sys.argv", argv),
            patch.object(_ep, "XpltParser", return_value=mock_parser_inst),
            patch.object(_ep, "FacetTracker", return_value=mock_tracker_inst),
        )

    def test_missing_file_exits_1(self, tmp_path):
        fake = tmp_path / "nonexistent.xplt"
        with patch("sys.argv", ["extract_pressure.py", str(fake)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1

    def test_bad_surface_name_exits_1(self, tmp_path):
        dummy = tmp_path / "dummy.xplt"
        dummy.write_bytes(b"\x00" * 4)
        mock_data = _make_mock_xplt_data(surface_name="RealSurface")
        series = _make_fake_series()

        p1, p2, p3 = self._patch_main(
            ["ep", str(dummy), "--surface", "FakeSurface", "--no-plot"],
            mock_data, series,
        )
        with p1, p2, p3:
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1

    def test_no_surfaces_exits_1(self, tmp_path):
        from digital_twin_ui.extraction.xplt_parser import XpltData
        dummy = tmp_path / "dummy.xplt"
        dummy.write_bytes(b"\x00" * 4)
        empty_data = XpltData()   # no surfaces

        mock_parser_inst = MagicMock()
        mock_parser_inst.parse.return_value = empty_data

        with (
            patch("sys.argv", ["ep", str(dummy), "--no-plot"]),
            patch.object(_ep, "XpltParser", return_value=mock_parser_inst),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1

    def test_no_plot_creates_csv_only(self, tmp_path):
        dummy = tmp_path / "dummy.xplt"
        dummy.write_bytes(b"\x00" * 4)
        mock_data = _make_mock_xplt_data()
        series = _make_fake_series(n_facets=3, n_states=2)
        out_dir = tmp_path / "out"

        p1, p2, p3 = self._patch_main(
            ["ep", str(dummy), "--surface", "TestSurface",
             "--no-plot", "--output-dir", str(out_dir)],
            mock_data, series,
        )
        with p1, p2, p3:
            main()

        csv_files = list(out_dir.glob("*.csv"))
        png_files = list(out_dir.glob("*.png"))
        assert len(csv_files) == 1
        assert len(png_files) == 0

    def test_default_surface_selected_when_not_given(self, tmp_path, capsys):
        dummy = tmp_path / "dummy.xplt"
        dummy.write_bytes(b"\x00" * 4)
        mock_data = _make_mock_xplt_data(surface_name="AutoSurface")
        series = _make_fake_series()
        out_dir = tmp_path / "out"

        p1, p2, p3 = self._patch_main(
            ["ep", str(dummy), "--no-plot", "--output-dir", str(out_dir)],
            mock_data, series,
        )
        with p1, p2, p3:
            main()

        captured = capsys.readouterr()
        assert "AutoSurface" in captured.out

    def test_csv_has_correct_name_stem(self, tmp_path):
        dummy = tmp_path / "my_run.xplt"
        dummy.write_bytes(b"\x00" * 4)
        mock_data = _make_mock_xplt_data(surface_name="TestSurface")
        series = _make_fake_series(n_facets=2, n_states=2)
        out_dir = tmp_path / "out"

        p1, p2, p3 = self._patch_main(
            ["ep", str(dummy), "--surface", "TestSurface",
             "--no-plot", "--output-dir", str(out_dir)],
            mock_data, series,
        )
        with p1, p2, p3:
            main()

        csv_files = list(out_dir.glob("*.csv"))
        assert len(csv_files) == 1
        assert "my_run" in csv_files[0].name
        assert "TestSurface" in csv_files[0].name

    def test_speed_stored_in_csv(self, tmp_path):
        dummy = tmp_path / "run.xplt"
        dummy.write_bytes(b"\x00" * 4)
        mock_data = _make_mock_xplt_data()
        series = _make_fake_series(speed=7.5)
        out_dir = tmp_path / "out"

        p1, p2, p3 = self._patch_main(
            ["ep", str(dummy), "--surface", "TestSurface", "--speed", "7.5",
             "--no-plot", "--output-dir", str(out_dir)],
            mock_data, series,
        )
        with p1, p2, p3:
            main()

        csv_file = next(out_dir.glob("*.csv"))
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert float(row["speed_mm_s"]) == pytest.approx(7.5)

    def test_plot_created_when_geometry_available(self, tmp_path):
        """Verify that plot_static is called (and not skipped) when geometry is present.
        We patch plot_static itself so we are not sensitive to tricontourf internals."""
        dummy = tmp_path / "run.xplt"
        dummy.write_bytes(b"\x00" * 4)
        mock_data = _make_mock_xplt_data(n_faces=15, n_nodes=9)
        series = _make_fake_series(n_facets=15, n_states=3)
        out_dir = tmp_path / "out"

        p1, p2, p3 = self._patch_main(
            ["ep", str(dummy), "--surface", "TestSurface",
             "--output-dir", str(out_dir)],
            mock_data, series,
        )
        with p1, p2, p3, patch.object(_ep, "plot_static") as mock_plot:
            main()

        mock_plot.assert_called_once()
        call_kwargs = mock_plot.call_args
        # output path argument is positional arg 7 (0-based)
        png_path = call_kwargs.args[7]
        assert str(png_path).endswith(".png")


# ===========================================================================
# Integration tests (require real xplt file)
# ===========================================================================

@pytest.mark.integration
class TestExtractPressureScriptIntegration:
    """End-to-end tests against the real sample xplt file."""

    @pytest.fixture(autouse=True)
    def require_sample(self):
        if not SAMPLE_XPLT.exists():
            pytest.skip(f"Sample xplt not found: {SAMPLE_XPLT}")

    def test_list_surfaces_prints_surface_names(self, capsys):
        list_surfaces(SAMPLE_XPLT)
        out = capsys.readouterr().out
        assert SAMPLE_SURFACE in out

    def test_save_csv_from_real_file(self, tmp_path):
        from digital_twin_ui.extraction.facet_tracker import FacetTracker
        tracker = FacetTracker()
        series = tracker.extract(
            SAMPLE_XPLT, speed_mm_s=5.0,
            surface_name=SAMPLE_SURFACE, facet_ids=list(range(5)),
        )
        out = tmp_path / "test_output.csv"
        save_csv(series, out)
        assert out.exists()
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        # 5 facets × 41 states
        assert len(rows) == 5 * SAMPLE_N_STATES
        assert "contact_pressure" in rows[0]
        assert "facet_area" in rows[0]

    def test_full_pipeline_no_plot(self, tmp_path):
        """main() with --no-plot against the real sample file."""
        with (
            patch("sys.argv", [
                "ep", str(SAMPLE_XPLT),
                "--surface", SAMPLE_SURFACE,
                "--speed", "5.0",
                "--no-plot",
                "--output-dir", str(tmp_path),
            ]),
        ):
            main()
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 1
        assert csv_files[0].stat().st_size > 0

    def test_full_pipeline_with_plot(self, tmp_path):
        """main() produces a PNG when run against the real sample file.
        We patch plot_static to avoid rendering 2734 faces in CI."""
        with (
            patch("sys.argv", [
                "ep", str(SAMPLE_XPLT),
                "--surface", SAMPLE_SURFACE,
                "--output-dir", str(tmp_path),
            ]),
            patch.object(_ep, "plot_static") as mock_plot,
        ):
            main()

        mock_plot.assert_called_once()
        # Confirm the output path passed to plot_static is a PNG
        png_path = mock_plot.call_args.args[7]
        assert str(png_path).endswith(".png")
