"""
Tests for CatheterCatalogue (simulation/catheter_catalogue.py).

Coverage
--------
- Loading the real catalogue YAML and verifying its structure
- Correct number of designs and their configurations
- get_design() lookup — success and KeyError
- get_configuration() lookup — success and KeyError
- resolve() returns a TemplateConfig pointing to base_configuration/
- resolve() raises FileNotFoundError for a missing .feb
- uniform_speeds() produces the correct length and value
- SimulationParams values match the YAML
- Singleton get_catalogue() reuses the same object
- Catalogue survives with a minimal in-memory YAML (no real files needed for unit tests)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CATALOGUE_PATH = PROJECT_ROOT / "config" / "catheter_catalogue.yaml"
BASE_CONFIG_DIR = PROJECT_ROOT / "base_configuration"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_catalogue(tmp_path: Path) -> Path:
    """Write a small in-memory catalogue YAML with two designs for fast unit tests."""
    data = {
        "designs": {
            "design_a": {
                "label": "Design A",
                "configurations": {
                    "14Fr_IR12": {"label": "14Fr IR12", "feb_file": "a_14fr_ir12.feb"},
                    "16Fr_IR12": {"label": "16Fr IR12", "feb_file": "a_16fr_ir12.feb"},
                },
            },
            "design_b": {
                "label": "Design B",
                "configurations": {
                    "14Fr_IR25": {"label": "14Fr IR25", "feb_file": "b_14fr_ir25.feb"},
                },
            },
        },
        "simulation": {
            "n_steps": 10,
            "base_step_size": 0.1,
            "default_dwell_time_s": 1.0,
            "displacements_mm": [64.0, 46.0] + [28.0] * 8,
            "speed_range": {"min_mm_s": 10.0, "max_mm_s": 25.0},
            "default_uniform_speed_mm_s": 15.0,
        },
    }
    path = tmp_path / "catheter_catalogue.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


def _make_catalogue(tmp_path: Path):
    """Return a CatheterCatalogue backed by the minimal in-memory YAML."""
    from digital_twin_ui.simulation.catheter_catalogue import CatheterCatalogue

    cat_path = _make_minimal_catalogue(tmp_path)
    return CatheterCatalogue(catalogue_path=cat_path, project_root=tmp_path)


# ===========================================================================
# Unit tests — use minimal in-memory catalogue (fast, no real files)
# ===========================================================================


class TestCatalogueLoading:
    def test_designs_count(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        assert len(cat.designs) == 2

    def test_design_names(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        names = [d.name for d in cat.designs]
        assert "design_a" in names
        assert "design_b" in names

    def test_design_labels(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        a = cat.get_design("design_a")
        assert a.label == "Design A"

    def test_configuration_count(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        assert len(cat.get_design("design_a").configurations) == 2
        assert len(cat.get_design("design_b").configurations) == 1

    def test_configuration_keys(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        keys = [c.key for c in cat.get_design("design_a").configurations]
        assert "14Fr_IR12" in keys
        assert "16Fr_IR12" in keys

    def test_configuration_feb_file(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        cfg = cat.get_design("design_a").get_configuration("14Fr_IR12")
        assert cfg.feb_file == "a_14fr_ir12.feb"

    def test_configuration_label(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        cfg = cat.get_design("design_a").get_configuration("16Fr_IR12")
        assert cfg.label == "16Fr IR12"


class TestSimulationParams:
    def test_n_steps(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        assert cat.simulation_params.n_steps == 10

    def test_base_step_size(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        assert cat.simulation_params.base_step_size == pytest.approx(0.1)

    def test_default_dwell(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        assert cat.simulation_params.default_dwell_time_s == pytest.approx(1.0)

    def test_displacements_length(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        assert len(cat.simulation_params.displacements_mm) == 10

    def test_displacements_values(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        d = cat.simulation_params.displacements_mm
        assert d[0] == pytest.approx(64.0)
        assert d[1] == pytest.approx(46.0)
        assert all(v == pytest.approx(28.0) for v in d[2:])

    def test_speed_range(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        p = cat.simulation_params
        assert p.speed_min_mm_s == pytest.approx(10.0)
        assert p.speed_max_mm_s == pytest.approx(25.0)

    def test_default_uniform_speed(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        assert cat.simulation_params.default_uniform_speed_mm_s == pytest.approx(15.0)


class TestLookupErrors:
    def test_unknown_design_raises_key_error(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            cat.get_design("nonexistent_design")

    def test_unknown_configuration_raises_key_error(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        design = cat.get_design("design_a")
        with pytest.raises(KeyError, match="not found"):
            design.get_configuration("99Fr_IR99")

    def test_resolve_missing_feb_raises_file_not_found(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        # The FEB files don't actually exist in tmp_path/base_configuration/
        with pytest.raises(FileNotFoundError, match="not found"):
            cat.resolve("design_a", "14Fr_IR12")

    def test_missing_catalogue_file_raises_file_not_found(self, tmp_path):
        from digital_twin_ui.simulation.catheter_catalogue import CatheterCatalogue

        missing = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError):
            CatheterCatalogue(catalogue_path=missing)


class TestResolve:
    def test_resolve_returns_template_config(self, tmp_path):
        """resolve() returns TemplateConfig with feb_path under base_configuration/."""
        from digital_twin_ui.simulation.catheter_catalogue import CatheterCatalogue
        from digital_twin_ui.simulation.template_registry import TemplateConfig

        cat_path = _make_minimal_catalogue(tmp_path)

        # Create a stub FEB file so FileNotFoundError is not raised
        base_dir = tmp_path / "base_configuration"
        base_dir.mkdir()
        stub = base_dir / "a_14fr_ir12.feb"
        stub.write_text("<?xml version='1.0'?><febio_spec/>")

        cat = CatheterCatalogue(catalogue_path=cat_path, project_root=tmp_path)
        tc = cat.resolve("design_a", "14Fr_IR12")

        assert isinstance(tc, TemplateConfig)
        assert tc.feb_path == stub
        assert tc.n_steps == 10
        assert tc.base_step_size == pytest.approx(0.1)

    def test_resolve_name_encodes_design_and_config(self, tmp_path):
        from digital_twin_ui.simulation.catheter_catalogue import CatheterCatalogue

        cat_path = _make_minimal_catalogue(tmp_path)
        base_dir = tmp_path / "base_configuration"
        base_dir.mkdir()
        (base_dir / "a_14fr_ir12.feb").write_text("<?xml version='1.0'?><febio_spec/>")

        cat = CatheterCatalogue(catalogue_path=cat_path, project_root=tmp_path)
        tc = cat.resolve("design_a", "14Fr_IR12")
        assert "design_a" in tc.name
        assert "14Fr_IR12" in tc.name

    def test_resolve_speed_range(self, tmp_path):
        from digital_twin_ui.simulation.catheter_catalogue import CatheterCatalogue

        cat_path = _make_minimal_catalogue(tmp_path)
        base_dir = tmp_path / "base_configuration"
        base_dir.mkdir()
        (base_dir / "a_14fr_ir12.feb").write_text("<?xml version='1.0'?><febio_spec/>")

        cat = CatheterCatalogue(catalogue_path=cat_path, project_root=tmp_path)
        tc = cat.resolve("design_a", "14Fr_IR12")
        assert tc.speed_range.min_mm_s == pytest.approx(10.0)
        assert tc.speed_range.max_mm_s == pytest.approx(25.0)


class TestUniformSpeeds:
    def test_default_uniform_speed(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        speeds = cat.uniform_speeds()
        assert len(speeds) == 10
        assert all(s == pytest.approx(15.0) for s in speeds)

    def test_custom_uniform_speed(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        speeds = cat.uniform_speeds(20.0)
        assert len(speeds) == 10
        assert all(s == pytest.approx(20.0) for s in speeds)

    def test_uniform_speeds_length_matches_n_steps(self, tmp_path):
        cat = _make_catalogue(tmp_path)
        speeds = cat.uniform_speeds()
        assert len(speeds) == cat.simulation_params.n_steps


# ===========================================================================
# Auto-discovery tests
# ===========================================================================


class TestParseFebFilename:
    """Unit tests for the _parse_feb_filename() helper."""

    def _parse(self, stem: str):
        from digital_twin_ui.simulation.catheter_catalogue import _parse_feb_filename
        return _parse_feb_filename(stem)

    def test_ball_tip_14fr_ir12(self):
        assert self._parse("ball_tip_14FR_ir12") == ("ball_tip", "14", "12")

    def test_ball_tip_14fr_ir25(self):
        assert self._parse("ball_tip_14FR_ir25") == ("ball_tip", "14", "25")

    def test_ball_tip_16fr_ir12(self):
        assert self._parse("ball_tip_16FR_ir12") == ("ball_tip", "16", "12")

    def test_nelaton_tip_lowercase_fr(self):
        assert self._parse("nelaton_tip_14Fr_ir12") == ("nelaton_tip", "14", "12")

    def test_nelaton_tip_16fr(self):
        assert self._parse("nelaton_tip_16Fr_ir12") == ("nelaton_tip", "16", "12")

    def test_vapro_introducer_with_extra_word(self):
        # vapro_introducer_14Fr_tip_ir12 has an extra '_tip' between Fr and ir
        assert self._parse("vapro_introducer_14Fr_tip_ir12") == ("vapro_introducer", "14", "12")

    def test_vapro_introducer_16fr(self):
        assert self._parse("vapro_introducer_16Fr_tip_ir12") == ("vapro_introducer", "16", "12")

    def test_unrecognized_returns_none(self):
        assert self._parse("sample_catheterization") is None

    def test_no_ir_suffix_returns_none(self):
        assert self._parse("ball_tip_14Fr") is None

    def test_future_design_key(self):
        # A new design added later should also parse correctly
        result = self._parse("tiemann_tip_14Fr_ir12")
        assert result == ("tiemann_tip", "14", "12")


class TestDesignLabelFromKey:
    def _label(self, key: str) -> str:
        from digital_twin_ui.simulation.catheter_catalogue import _design_label_from_key
        return _design_label_from_key(key)

    def test_ball_tip(self):
        assert self._label("ball_tip") == "Ball Tip"

    def test_nelaton_tip(self):
        assert self._label("nelaton_tip") == "Nelaton Tip"

    def test_vapro_introducer(self):
        assert self._label("vapro_introducer") == "Vapro Introducer"

    def test_single_word(self):
        assert self._label("tiemann") == "Tiemann"


class TestAutoDiscover:
    """Tests for CatheterCatalogue._auto_discover() via a real directory scan."""

    def _make_catalogue_with_dir(self, tmp_path: Path, feb_files: list[str]):
        """Create a minimal catalogue YAML and populate base_configuration/ with stub FEBs."""
        import yaml
        from digital_twin_ui.simulation.catheter_catalogue import CatheterCatalogue

        # Minimal YAML with one design (design_a / 14Fr_IR12 only)
        data = {
            "designs": {
                "design_a": {
                    "label": "Design A",
                    "configurations": {
                        "14Fr_IR12": {"label": "14Fr IR12", "feb_file": "design_a_14Fr_ir12.feb"},
                    },
                },
            },
            "simulation": {
                "n_steps": 10,
                "base_step_size": 0.1,
                "default_dwell_time_s": 1.0,
                "displacements_mm": [64.0, 46.0] + [28.0] * 8,
                "speed_range": {"min_mm_s": 10.0, "max_mm_s": 25.0},
                "default_uniform_speed_mm_s": 15.0,
            },
        }
        cat_path = tmp_path / "catheter_catalogue.yaml"
        cat_path.write_text(yaml.dump(data), encoding="utf-8")

        base_dir = tmp_path / "base_configuration"
        base_dir.mkdir()
        stub_content = "<?xml version='1.0'?><febio_spec/>"
        for fname in feb_files:
            (base_dir / fname).write_text(stub_content)

        return CatheterCatalogue(catalogue_path=cat_path, project_root=tmp_path)

    def test_already_registered_not_duplicated(self, tmp_path):
        """A .feb file listed in the YAML must not appear twice."""
        cat = self._make_catalogue_with_dir(
            tmp_path, ["design_a_14Fr_ir12.feb"]
        )
        d = cat.get_design("design_a")
        assert len(d.configurations) == 1

    def test_new_config_added_to_existing_design(self, tmp_path):
        """A new size/IR variant for an existing design is registered automatically."""
        cat = self._make_catalogue_with_dir(
            tmp_path,
            ["design_a_14Fr_ir12.feb", "design_a_16Fr_ir25.feb"],
        )
        d = cat.get_design("design_a")
        keys = {c.key for c in d.configurations}
        assert "14Fr_IR12" in keys
        assert "16Fr_IR25" in keys

    def test_new_design_created(self, tmp_path):
        """A .feb file for an entirely new design creates a new CatalogueDesign."""
        cat = self._make_catalogue_with_dir(
            tmp_path,
            ["design_a_14Fr_ir12.feb", "tiemann_tip_14Fr_ir12.feb"],
        )
        names = {d.name for d in cat.designs}
        assert "tiemann_tip" in names

    def test_new_design_label_generated(self, tmp_path):
        """Auto-discovered designs get a title-case label from their key."""
        cat = self._make_catalogue_with_dir(
            tmp_path, ["design_a_14Fr_ir12.feb", "tiemann_tip_16Fr_ir25.feb"]
        )
        d = cat.get_design("tiemann_tip")
        assert d.label == "Tiemann Tip"

    def test_unrecognized_file_skipped(self, tmp_path):
        """Files that don't match the naming convention are silently ignored."""
        cat = self._make_catalogue_with_dir(
            tmp_path, ["design_a_14Fr_ir12.feb", "sample_catheterization.feb"]
        )
        assert len(cat.designs) == 1

    def test_no_base_config_dir(self, tmp_path):
        """If base_configuration/ doesn't exist, _auto_discover is a no-op."""
        import yaml
        from digital_twin_ui.simulation.catheter_catalogue import CatheterCatalogue

        data = {
            "designs": {},
            "simulation": {
                "n_steps": 10, "base_step_size": 0.1, "default_dwell_time_s": 1.0,
                "displacements_mm": [28.0] * 10,
                "speed_range": {"min_mm_s": 10.0, "max_mm_s": 25.0},
                "default_uniform_speed_mm_s": 15.0,
            },
        }
        cat_path = tmp_path / "cat.yaml"
        cat_path.write_text(yaml.dump(data))
        cat = CatheterCatalogue(catalogue_path=cat_path, project_root=tmp_path)
        assert cat.designs == []

    def test_auto_discovered_config_feb_file_name(self, tmp_path):
        """The registered feb_file must match the actual filename on disk."""
        cat = self._make_catalogue_with_dir(
            tmp_path, ["design_a_14Fr_ir12.feb", "tiemann_tip_16Fr_ir12.feb"]
        )
        d = cat.get_design("tiemann_tip")
        cfg = d.get_configuration("16Fr_IR12")
        assert cfg.feb_file == "tiemann_tip_16Fr_ir12.feb"

    def test_vapro_introducer_tip_in_name(self, tmp_path):
        """Files with extra words between Fr and ir (like vapro_introducer_14Fr_tip_ir12) parse correctly."""
        cat = self._make_catalogue_with_dir(
            tmp_path,
            ["design_a_14Fr_ir12.feb", "vapro_introducer_14Fr_tip_ir12.feb"],
        )
        names = {d.name for d in cat.designs}
        assert "vapro_introducer" in names
        d = cat.get_design("vapro_introducer")
        assert d.get_configuration("14Fr_IR12").feb_file == "vapro_introducer_14Fr_tip_ir12.feb"


class TestResetSingleton:
    def test_reset_causes_reload(self, tmp_path):
        from digital_twin_ui.simulation.catheter_catalogue import (
            get_catalogue,
            reset_catalogue_singleton,
        )

        reset_catalogue_singleton()
        cat1 = get_catalogue()
        reset_catalogue_singleton()
        cat2 = get_catalogue()
        # After reset a new object is created
        assert cat1 is not cat2

    def test_same_object_without_reset(self):
        from digital_twin_ui.simulation.catheter_catalogue import get_catalogue

        cat1 = get_catalogue()
        cat2 = get_catalogue()
        assert cat1 is cat2


# ===========================================================================
# Integration tests — validate the real catalogue + base_configuration/ files
# ===========================================================================


@pytest.mark.integration
class TestRealCatalogue:
    """Validate the real catheter_catalogue.yaml and referenced FEB files."""

    @pytest.fixture(scope="class")
    def cat(self):
        if not CATALOGUE_PATH.exists():
            pytest.skip(f"Catalogue not found: {CATALOGUE_PATH}")
        from digital_twin_ui.simulation.catheter_catalogue import CatheterCatalogue
        return CatheterCatalogue(catalogue_path=CATALOGUE_PATH, project_root=PROJECT_ROOT)

    def test_exactly_three_designs(self, cat):
        assert len(cat.designs) == 3

    def test_design_names_present(self, cat):
        names = {d.name for d in cat.designs}
        assert names == {"ball_tip", "nelaton_tip", "vapro_introducer"}

    def test_ball_tip_configurations(self, cat):
        d = cat.get_design("ball_tip")
        keys = {c.key for c in d.configurations}
        assert "14Fr_IR12" in keys
        assert "14Fr_IR25" in keys
        assert "16Fr_IR12" in keys

    def test_nelaton_tip_configurations(self, cat):
        d = cat.get_design("nelaton_tip")
        keys = {c.key for c in d.configurations}
        assert "14Fr_IR12" in keys
        assert "14Fr_IR25" in keys
        assert "16Fr_IR12" in keys

    def test_vapro_introducer_configurations(self, cat):
        d = cat.get_design("vapro_introducer")
        keys = {c.key for c in d.configurations}
        assert "14Fr_IR12" in keys
        assert "16Fr_IR12" in keys

    def test_all_feb_files_exist(self, cat):
        """Every FEB file listed in the catalogue must exist in base_configuration/."""
        missing = []
        for design in cat.designs:
            for cfg in design.configurations:
                feb = BASE_CONFIG_DIR / cfg.feb_file
                if not feb.exists():
                    missing.append(str(feb))
        assert not missing, f"Missing FEB files:\n" + "\n".join(missing)

    def test_n_steps_is_ten(self, cat):
        assert cat.simulation_params.n_steps == 10

    def test_displacements_length_matches_n_steps(self, cat):
        p = cat.simulation_params
        assert len(p.displacements_mm) == p.n_steps

    def test_all_displacements_positive(self, cat):
        assert all(d > 0 for d in cat.simulation_params.displacements_mm)

    def test_speed_range_valid(self, cat):
        p = cat.simulation_params
        assert p.speed_min_mm_s < p.speed_max_mm_s
        assert p.speed_min_mm_s > 0

    def test_resolve_all_configurations(self, cat):
        """resolve() must succeed for every design+config in the catalogue."""
        failed = []
        for design in cat.designs:
            for cfg in design.configurations:
                try:
                    tc = cat.resolve(design.name, cfg.key)
                    assert tc.feb_path.exists(), f"FEB missing: {tc.feb_path}"
                except Exception as exc:
                    failed.append(f"{design.name}/{cfg.key}: {exc}")
        assert not failed, "\n".join(failed)
