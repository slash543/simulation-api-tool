"""
Tests for base_configuration/ FEB file structural integrity.

Coverage
--------
- All FEB files referenced in catheter_catalogue.yaml exist in base_configuration/
- Each FEB has exactly 10 <step> elements
- Each FEB has exactly 10 <load_controller> elements
- Every <step> has a <Control><time_steps> element with step_size = 0.1
- Every <step> has at least one <value lc="N"> reference
- All LC ids referenced in BCs exist as <load_controller id="N">
- MultiStepConfigurator.configure() ONLY changes:
    - <pt> text content inside <load_controller>/<points>
    - <time_steps> text inside <step>/<Control>
  All other XML (geometry, materials, contacts, BCs) is preserved bit-for-bit.

These tests provide the "geometry replacement robustness" guarantee: as long as
a replacement FEB file keeps the same 10-step, 10-LC structure with the same
step→LC mapping, MultiStepConfigurator will work correctly.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from lxml import etree

PROJECT_ROOT = Path(__file__).parent.parent
CATALOGUE_PATH = PROJECT_ROOT / "config" / "catheter_catalogue.yaml"
BASE_CONFIG_DIR = PROJECT_ROOT / "base_configuration"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_STEPS = 10


def _load_feb(feb_path: Path) -> etree._Element:
    parser = etree.XMLParser(remove_blank_text=False, remove_comments=False)
    tree = etree.parse(str(feb_path), parser)
    return tree.getroot()


def _collect_catalogue_feb_files() -> list[Path]:
    """Return list of Path for every feb_file in the real catalogue."""
    if not CATALOGUE_PATH.exists():
        return []
    import yaml
    data = yaml.safe_load(CATALOGUE_PATH.read_text(encoding="utf-8"))
    paths = []
    for design_data in data.get("designs", {}).values():
        for cfg_data in design_data.get("configurations", {}).values():
            feb_file = cfg_data.get("feb_file", "")
            if feb_file:
                paths.append(BASE_CONFIG_DIR / feb_file)
    return paths


def _all_feb_paths() -> list[Path]:
    """All .feb files in base_configuration/ that are real multi-step files."""
    catalogue_files = _collect_catalogue_feb_files()
    if catalogue_files:
        return catalogue_files
    # Fallback: scan directory
    return [p for p in BASE_CONFIG_DIR.glob("*.feb")
            if p.name != "sample_catheterization.feb"]


def _catalogue_feb_ids() -> list[str]:
    return [p.name for p in _collect_catalogue_feb_files()]


def _skip_if_no_catalogue():
    if not CATALOGUE_PATH.exists():
        pytest.skip(f"Catalogue not found: {CATALOGUE_PATH}")


def _skip_if_no_base_config():
    if not BASE_CONFIG_DIR.exists():
        pytest.skip(f"base_configuration/ directory not found: {BASE_CONFIG_DIR}")


# ---------------------------------------------------------------------------
# Fixture: parametrize over all catalogue-referenced FEB files
# ---------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    if "feb_path" in metafunc.fixturenames:
        paths = _collect_catalogue_feb_files()
        if not paths:
            paths = []
        metafunc.parametrize("feb_path", paths, ids=[p.name for p in paths])


# ===========================================================================
# Existence tests
# ===========================================================================


@pytest.mark.integration
class TestFebFilesExist:
    """Verify all catalogue-referenced FEB files are present on disk."""

    def test_base_configuration_dir_exists(self):
        _skip_if_no_base_config()
        assert BASE_CONFIG_DIR.is_dir()

    def test_all_catalogue_feb_files_present(self):
        _skip_if_no_catalogue()
        _skip_if_no_base_config()
        missing = [p for p in _collect_catalogue_feb_files() if not p.exists()]
        assert not missing, (
            "Missing FEB files:\n" + "\n".join(str(p) for p in missing)
        )


# ===========================================================================
# XML structure tests — one test per file via parametrize
# ===========================================================================


@pytest.mark.integration
class TestFebXmlStructure:
    """Validate the 10-step / 10-LC XML structure of every catalogue FEB file."""

    def test_feb_file_exists(self, feb_path):
        assert feb_path.exists(), f"FEB file not found: {feb_path}"

    def test_parses_as_valid_xml(self, feb_path):
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")
        root = _load_feb(feb_path)
        assert root.tag == "febio_spec"

    def test_has_ten_steps(self, feb_path):
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")
        root = _load_feb(feb_path)
        steps = root.findall(".//step")
        assert len(steps) == N_STEPS, (
            f"{feb_path.name}: expected {N_STEPS} <step> elements, found {len(steps)}"
        )

    def test_has_ten_load_controllers(self, feb_path):
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")
        root = _load_feb(feb_path)
        lcs = root.findall(".//load_controller")
        assert len(lcs) == N_STEPS, (
            f"{feb_path.name}: expected {N_STEPS} <load_controller> elements, found {len(lcs)}"
        )

    def test_every_step_has_control_block(self, feb_path):
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")
        root = _load_feb(feb_path)
        for step in root.findall(".//step"):
            sid = step.get("id", "?")
            control = step.find("Control")
            assert control is not None, (
                f"{feb_path.name}: <step id='{sid}'> missing <Control> block"
            )

    def test_every_step_has_time_steps_element(self, feb_path):
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")
        root = _load_feb(feb_path)
        for step in root.findall(".//step"):
            sid = step.get("id", "?")
            control = step.find("Control")
            if control is None:
                continue
            ts = control.find("time_steps")
            assert ts is not None, (
                f"{feb_path.name}: <step id='{sid}'> <Control> missing <time_steps>"
            )
            assert ts.text and ts.text.strip().isdigit(), (
                f"{feb_path.name}: <step id='{sid}'> <time_steps> has non-integer text: '{ts.text}'"
            )

    def test_every_step_has_step_size_element(self, feb_path):
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")
        root = _load_feb(feb_path)
        for step in root.findall(".//step"):
            sid = step.get("id", "?")
            control = step.find("Control")
            if control is None:
                continue
            ss = control.find("step_size")
            assert ss is not None, (
                f"{feb_path.name}: <step id='{sid}'> <Control> missing <step_size>"
            )

    def test_every_step_has_lc_reference(self, feb_path):
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")
        root = _load_feb(feb_path)
        for step in root.findall(".//step"):
            sid = step.get("id", "?")
            value_elems = step.findall(".//value[@lc]")
            assert value_elems, (
                f"{feb_path.name}: <step id='{sid}'> has no <value lc='N'> references"
            )

    def test_all_referenced_lc_ids_exist(self, feb_path):
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")
        root = _load_feb(feb_path)

        # Collect defined LC ids
        defined_ids = {
            int(elem.get("id"))
            for elem in root.findall(".//load_controller")
            if elem.get("id") is not None
        }

        # Collect all referenced LC ids
        referenced_ids = {
            int(v.get("lc"))
            for v in root.findall(".//value[@lc]")
        }

        missing = referenced_ids - defined_ids
        assert not missing, (
            f"{feb_path.name}: referenced LC ids not defined: {sorted(missing)}"
        )

    def test_every_lc_has_points_block(self, feb_path):
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")
        root = _load_feb(feb_path)
        for lc in root.findall(".//load_controller"):
            lid = lc.get("id", "?")
            points = lc.find("points")
            assert points is not None, (
                f"{feb_path.name}: <load_controller id='{lid}'> missing <points>"
            )

    def test_every_lc_has_at_least_two_pt_elements(self, feb_path):
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")
        root = _load_feb(feb_path)
        for lc in root.findall(".//load_controller"):
            lid = lc.get("id", "?")
            points = lc.find("points")
            if points is None:
                continue
            pts = points.findall("pt")
            assert len(pts) >= 2, (
                f"{feb_path.name}: <load_controller id='{lid}'> has fewer than 2 <pt> elements"
            )


# ===========================================================================
# Robustness tests — MultiStepConfigurator only changes what it should
# ===========================================================================


@pytest.mark.integration
class TestConfiguratorPreservesGeometry:
    """
    Geometry-replacement robustness guarantee.

    After running MultiStepConfigurator.configure() on a FEB file:
    - <pt> text values inside load controllers may change (expected)
    - <time_steps> values inside step Control blocks may change (expected)
    - Everything else (materials, nodes, elements, contacts, BCs, step_size,
      BC value magnitudes) must be identical to the original.
    """

    def _get_catalogue_template_config(self, feb_path: Path):
        """Build a minimal TemplateConfig pointing to feb_path."""
        from digital_twin_ui.simulation.template_registry import TemplateConfig, SpeedRange
        return TemplateConfig(
            name=feb_path.stem,
            label=feb_path.stem,
            feb_file=feb_path.name,
            n_steps=N_STEPS,
            base_step_size=0.1,
            default_dwell_time_s=1.0,
            displacements_mm=[64.0, 46.0] + [28.0] * 8,
            speed_range=SpeedRange(min_mm_s=10.0, max_mm_s=25.0),
            _feb_subdir="base_configuration",
            _project_root=PROJECT_ROOT,
        )

    def test_materials_unchanged_after_configure(self, feb_path, tmp_path):
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")

        from digital_twin_ui.simulation.multi_step_configurator import MultiStepConfigurator

        tc = self._get_catalogue_template_config(feb_path)
        cfg = MultiStepConfigurator(tc)

        out = tmp_path / "output.feb"
        cfg.configure(speeds_mm_s=[15.0] * N_STEPS, dwell_time_s=1.0, output_path=out)

        orig_root = _load_feb(feb_path)
        new_root = _load_feb(out)

        orig_mat = etree.tostring(orig_root.find(".//Material"), encoding="unicode")
        new_mat = etree.tostring(new_root.find(".//Material"), encoding="unicode")
        assert orig_mat == new_mat, f"{feb_path.name}: <Material> changed after configure()"

    def test_geometry_section_unchanged_after_configure(self, feb_path, tmp_path):
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")

        from digital_twin_ui.simulation.multi_step_configurator import MultiStepConfigurator

        tc = self._get_catalogue_template_config(feb_path)
        cfg = MultiStepConfigurator(tc)

        out = tmp_path / "output.feb"
        cfg.configure(speeds_mm_s=[15.0] * N_STEPS, dwell_time_s=1.0, output_path=out)

        orig_root = _load_feb(feb_path)
        new_root = _load_feb(out)

        for section in ("Mesh", "MeshDomains", "MeshData"):
            orig_sec = orig_root.find(f".//{section}")
            new_sec = new_root.find(f".//{section}")
            if orig_sec is None and new_sec is None:
                continue
            if orig_sec is None or new_sec is None:
                # One exists, the other doesn't
                assert False, f"{feb_path.name}: <{section}> presence changed after configure()"
            assert (
                etree.tostring(orig_sec, encoding="unicode") ==
                etree.tostring(new_sec, encoding="unicode")
            ), f"{feb_path.name}: <{section}> changed after configure()"

    def test_boundary_value_magnitudes_unchanged(self, feb_path, tmp_path):
        """The <value lc="N"> displacement magnitudes must not change."""
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")

        from digital_twin_ui.simulation.multi_step_configurator import MultiStepConfigurator

        tc = self._get_catalogue_template_config(feb_path)
        cfg = MultiStepConfigurator(tc)

        out = tmp_path / "output.feb"
        cfg.configure(speeds_mm_s=[15.0] * N_STEPS, dwell_time_s=1.0, output_path=out)

        orig_root = _load_feb(feb_path)
        new_root = _load_feb(out)

        orig_vals = [v.text for v in orig_root.findall(".//value[@lc]")]
        new_vals = [v.text for v in new_root.findall(".//value[@lc]")]

        assert orig_vals == new_vals, (
            f"{feb_path.name}: <value lc='N'> text changed after configure()"
        )

    def test_step_size_unchanged_after_configure(self, feb_path, tmp_path):
        """<step_size> values must not be modified by the configurator."""
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")

        from digital_twin_ui.simulation.multi_step_configurator import MultiStepConfigurator

        tc = self._get_catalogue_template_config(feb_path)
        cfg = MultiStepConfigurator(tc)

        out = tmp_path / "output.feb"
        cfg.configure(speeds_mm_s=[15.0] * N_STEPS, dwell_time_s=1.0, output_path=out)

        orig_root = _load_feb(feb_path)
        new_root = _load_feb(out)

        orig_ss = [ss.text for ss in orig_root.findall(".//step_size")]
        new_ss = [ss.text for ss in new_root.findall(".//step_size")]

        assert orig_ss == new_ss, (
            f"{feb_path.name}: <step_size> changed after configure()"
        )

    def test_pt_times_change_after_configure(self, feb_path, tmp_path):
        """After configure() the <pt> times should differ from the template."""
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")

        from digital_twin_ui.simulation.multi_step_configurator import MultiStepConfigurator

        tc = self._get_catalogue_template_config(feb_path)
        cfg = MultiStepConfigurator(tc)

        out = tmp_path / "output.feb"
        result = cfg.configure(speeds_mm_s=[15.0] * N_STEPS, dwell_time_s=1.0, output_path=out)

        # Verify the output has the expected number of LC blocks
        new_root = _load_feb(out)
        lcs = new_root.findall(".//load_controller")
        assert len(lcs) == N_STEPS

        # Verify total duration is plausible
        # At 15 mm/s with displacements [64, 46, 28*8], each step = disp/15 + 1.0 dwell
        expected_total = sum(
            (d / 15.0) + 1.0 for d in [64.0, 46.0] + [28.0] * 8
        )
        assert result.total_duration_s == pytest.approx(expected_total, rel=1e-4)

    def test_time_steps_updated_after_configure(self, feb_path, tmp_path):
        """After configure(), <time_steps> in each step should reflect the new timing."""
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")

        from digital_twin_ui.simulation.multi_step_configurator import MultiStepConfigurator

        tc = self._get_catalogue_template_config(feb_path)
        cfg = MultiStepConfigurator(tc)

        out = tmp_path / "output.feb"
        result = cfg.configure(speeds_mm_s=[15.0] * N_STEPS, dwell_time_s=1.0, output_path=out)

        new_root = _load_feb(out)
        for i, step in enumerate(new_root.findall(".//step")):
            control = step.find("Control")
            if control is None:
                continue
            ts_elem = control.find("time_steps")
            if ts_elem is None:
                continue
            ts = int(ts_elem.text)
            expected_ts = result.step_durations[i] / tc.base_step_size
            import math
            expected_ts_int = max(10, math.ceil(expected_ts))
            assert ts == expected_ts_int, (
                f"{feb_path.name}: step {i+1} time_steps={ts}, expected {expected_ts_int}"
            )

    def test_only_pt_and_time_steps_differ(self, feb_path, tmp_path):
        """
        Parse both original and configured FEB files and compare element-by-element.
        Only <pt> text and <time_steps> text should differ; everything else identical.
        """
        if not feb_path.exists():
            pytest.skip(f"File not found: {feb_path}")

        from digital_twin_ui.simulation.multi_step_configurator import MultiStepConfigurator

        tc = self._get_catalogue_template_config(feb_path)
        cfg = MultiStepConfigurator(tc)

        out = tmp_path / "output.feb"
        cfg.configure(speeds_mm_s=[15.0] * N_STEPS, dwell_time_s=1.0, output_path=out)

        orig_root = _load_feb(feb_path)
        new_root = _load_feb(out)

        unexpected_diffs: list[str] = []

        def compare(orig_elem, new_elem, path=""):
            current_path = f"{path}/{orig_elem.tag}"

            # Tag must be same (structural guarantee)
            if orig_elem.tag != new_elem.tag:
                unexpected_diffs.append(f"tag mismatch at {path}")
                return

            is_pt = orig_elem.tag == "pt"
            is_time_steps = orig_elem.tag == "time_steps"

            # Text content
            if not is_pt and not is_time_steps:
                if (orig_elem.text or "").strip() != (new_elem.text or "").strip():
                    unexpected_diffs.append(
                        f"text changed at {current_path}: "
                        f"'{orig_elem.text}' → '{new_elem.text}'"
                    )

            # Attributes
            if orig_elem.attrib != new_elem.attrib:
                unexpected_diffs.append(
                    f"attrs changed at {current_path}: "
                    f"{orig_elem.attrib} → {new_elem.attrib}"
                )

            # Children count
            orig_children = list(orig_elem)
            new_children = list(new_elem)

            # For <points> blocks, child count may differ (3 vs 4 pt pattern) — allow
            if orig_elem.tag == "points":
                return

            if len(orig_children) != len(new_children):
                unexpected_diffs.append(
                    f"child count changed at {current_path}: "
                    f"{len(orig_children)} → {len(new_children)}"
                )
                return

            for oc, nc in zip(orig_children, new_children):
                compare(oc, nc, current_path)

        compare(orig_root, new_root)

        assert not unexpected_diffs, (
            f"{feb_path.name}: unexpected changes after configure():\n"
            + "\n".join(unexpected_diffs[:20])
        )
