"""
Tests for TemplateRegistry and TemplateConfig (simulation/template_registry.py).

Coverage
--------
- Registry scans YAML files correctly
- TemplateConfig fields parsed from YAML
- SpeedRange min/max values
- feb_path property resolves correctly
- get() raises KeyError for unknown names
- list_templates() returns sorted names
- all_configs() returns all configs
- is_multi_step property (n_steps > 1)
- as_dict() serialisation
- Missing / empty templates directory handled gracefully
- get_template_registry() singleton
- Real YAML templates loaded (integration)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from digital_twin_ui.simulation.template_registry import (
    SpeedRange,
    TemplateConfig,
    TemplateRegistry,
    get_template_registry,
)


# ---------------------------------------------------------------------------
# Helpers — build minimal YAML files for unit tests
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


def _sample_yaml_data(name: str = "test_template", n_steps: int = 10) -> dict:
    return {
        "name": name,
        "label": f"{name} label",
        "feb_file": f"{name}.feb",
        "n_steps": n_steps,
        "base_step_size": 0.1,
        "default_dwell_time_s": 1.0,
        "displacements_mm": [10.0] * n_steps,
        "speed_range": {"min_mm_s": 5.0, "max_mm_s": 30.0},
    }


@pytest.fixture()
def tmp_templates_dir(tmp_path: Path) -> Path:
    d = tmp_path / "config" / "templates"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def single_template_dir(tmp_templates_dir: Path) -> Path:
    _write_yaml(
        tmp_templates_dir / "my_template.yaml",
        _sample_yaml_data("my_template", n_steps=10),
    )
    return tmp_templates_dir


@pytest.fixture()
def multi_template_dir(tmp_templates_dir: Path) -> Path:
    for name, n in [("alpha", 1), ("beta", 10), ("gamma", 5)]:
        _write_yaml(
            tmp_templates_dir / f"{name}.yaml",
            _sample_yaml_data(name, n_steps=n),
        )
    return tmp_templates_dir


# ---------------------------------------------------------------------------
# TemplateConfig unit tests
# ---------------------------------------------------------------------------

class TestTemplateConfig:
    def test_is_multi_step_true(self) -> None:
        tc = TemplateConfig(
            name="t",
            label="T",
            feb_file="t.feb",
            n_steps=10,
            base_step_size=0.1,
            default_dwell_time_s=1.0,
            displacements_mm=[10.0] * 10,
            speed_range=SpeedRange(5.0, 30.0),
        )
        assert tc.is_multi_step is True

    def test_is_multi_step_false_for_single(self) -> None:
        tc = TemplateConfig(
            name="t",
            label="T",
            feb_file="t.feb",
            n_steps=1,
            base_step_size=0.05,
            default_dwell_time_s=0.0,
            displacements_mm=[10.0],
            speed_range=SpeedRange(4.0, 6.0),
        )
        assert tc.is_multi_step is False

    def test_feb_path_resolves_under_project_root(self, tmp_path: Path) -> None:
        tc = TemplateConfig(
            name="t",
            label="T",
            feb_file="t.feb",
            n_steps=10,
            base_step_size=0.1,
            default_dwell_time_s=1.0,
            displacements_mm=[10.0] * 10,
            speed_range=SpeedRange(5.0, 30.0),
            _project_root=tmp_path,
        )
        assert tc.feb_path == tmp_path / "templates" / "t.feb"

    def test_as_dict_keys(self) -> None:
        tc = TemplateConfig(
            name="x",
            label="X",
            feb_file="x.feb",
            n_steps=5,
            base_step_size=0.05,
            default_dwell_time_s=1.0,
            displacements_mm=[10.0] * 5,
            speed_range=SpeedRange(2.0, 20.0),
        )
        d = tc.as_dict()
        for key in ("name", "label", "feb_file", "n_steps", "base_step_size",
                    "default_dwell_time_s", "displacements_mm", "speed_range",
                    "is_multi_step", "feb_path"):
            assert key in d, f"Missing key: {key}"

    def test_as_dict_values(self) -> None:
        tc = TemplateConfig(
            name="x",
            label="X label",
            feb_file="x.feb",
            n_steps=3,
            base_step_size=0.1,
            default_dwell_time_s=2.0,
            displacements_mm=[5.0, 5.0, 5.0],
            speed_range=SpeedRange(10.0, 25.0),
        )
        d = tc.as_dict()
        assert d["name"] == "x"
        assert d["n_steps"] == 3
        assert d["speed_range"]["min_mm_s"] == 10.0
        assert d["speed_range"]["max_mm_s"] == 25.0
        assert d["displacements_mm"] == [5.0, 5.0, 5.0]

    def test_speed_range_fields(self) -> None:
        sr = SpeedRange(min_mm_s=10.0, max_mm_s=25.0)
        assert sr.min_mm_s == 10.0
        assert sr.max_mm_s == 25.0


# ---------------------------------------------------------------------------
# TemplateRegistry unit tests
# ---------------------------------------------------------------------------

class TestTemplateRegistryLoading:
    def test_loads_single_template(self, single_template_dir: Path, tmp_path: Path) -> None:
        reg = TemplateRegistry(templates_dir=single_template_dir, project_root=tmp_path)
        assert "my_template" in reg.list_templates()

    def test_loads_multiple_templates(self, multi_template_dir: Path, tmp_path: Path) -> None:
        reg = TemplateRegistry(templates_dir=multi_template_dir, project_root=tmp_path)
        names = reg.list_templates()
        assert set(names) == {"alpha", "beta", "gamma"}

    def test_list_templates_sorted(self, multi_template_dir: Path, tmp_path: Path) -> None:
        reg = TemplateRegistry(templates_dir=multi_template_dir, project_root=tmp_path)
        names = reg.list_templates()
        assert names == sorted(names)

    def test_all_configs_returns_all(self, multi_template_dir: Path, tmp_path: Path) -> None:
        reg = TemplateRegistry(templates_dir=multi_template_dir, project_root=tmp_path)
        configs = reg.all_configs()
        assert len(configs) == 3
        assert all(isinstance(c, TemplateConfig) for c in configs)

    def test_missing_directory_no_crash(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "does_not_exist"
        reg = TemplateRegistry(templates_dir=non_existent, project_root=tmp_path)
        assert reg.list_templates() == []

    def test_empty_directory_no_crash(self, tmp_templates_dir: Path, tmp_path: Path) -> None:
        reg = TemplateRegistry(templates_dir=tmp_templates_dir, project_root=tmp_path)
        assert reg.list_templates() == []

    def test_broken_yaml_skipped(self, tmp_templates_dir: Path, tmp_path: Path) -> None:
        good = tmp_templates_dir / "good.yaml"
        _write_yaml(good, _sample_yaml_data("good"))
        bad = tmp_templates_dir / "bad.yaml"
        bad.write_text("{{invalid: yaml: :", encoding="utf-8")
        reg = TemplateRegistry(templates_dir=tmp_templates_dir, project_root=tmp_path)
        # Should load good, skip bad
        assert "good" in reg.list_templates()
        assert "bad" not in reg.list_templates()


class TestTemplateRegistryGet:
    def test_get_returns_correct_config(self, single_template_dir: Path, tmp_path: Path) -> None:
        reg = TemplateRegistry(templates_dir=single_template_dir, project_root=tmp_path)
        tc = reg.get("my_template")
        assert tc.name == "my_template"
        assert tc.n_steps == 10
        assert tc.speed_range.min_mm_s == 5.0
        assert tc.speed_range.max_mm_s == 30.0

    def test_get_raises_key_error_for_unknown(self, single_template_dir: Path, tmp_path: Path) -> None:
        reg = TemplateRegistry(templates_dir=single_template_dir, project_root=tmp_path)
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent_template")

    def test_get_error_lists_available(self, multi_template_dir: Path, tmp_path: Path) -> None:
        reg = TemplateRegistry(templates_dir=multi_template_dir, project_root=tmp_path)
        with pytest.raises(KeyError) as exc_info:
            reg.get("missing")
        assert "alpha" in str(exc_info.value) or "beta" in str(exc_info.value)

    def test_config_displacements_parsed(self, tmp_templates_dir: Path, tmp_path: Path) -> None:
        data = _sample_yaml_data("t", n_steps=3)
        data["displacements_mm"] = [64.0, 46.0, 28.0]
        _write_yaml(tmp_templates_dir / "t.yaml", data)
        reg = TemplateRegistry(templates_dir=tmp_templates_dir, project_root=tmp_path)
        tc = reg.get("t")
        assert tc.displacements_mm == [64.0, 46.0, 28.0]

    def test_config_base_step_size_parsed(self, tmp_templates_dir: Path, tmp_path: Path) -> None:
        data = _sample_yaml_data("t")
        data["base_step_size"] = 0.025
        _write_yaml(tmp_templates_dir / "t.yaml", data)
        reg = TemplateRegistry(templates_dir=tmp_templates_dir, project_root=tmp_path)
        tc = reg.get("t")
        assert tc.base_step_size == pytest.approx(0.025)

    def test_feb_path_uses_project_root(self, single_template_dir: Path, tmp_path: Path) -> None:
        reg = TemplateRegistry(templates_dir=single_template_dir, project_root=tmp_path)
        tc = reg.get("my_template")
        assert tc.feb_path == tmp_path / "templates" / "my_template.feb"


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

class TestGetTemplateRegistry:
    def test_returns_registry_instance(self) -> None:
        import digital_twin_ui.simulation.template_registry as mod
        mod._registry_singleton = None  # reset
        reg = get_template_registry()
        assert isinstance(reg, TemplateRegistry)

    def test_singleton_cached(self) -> None:
        import digital_twin_ui.simulation.template_registry as mod
        mod._registry_singleton = None
        r1 = get_template_registry()
        r2 = get_template_registry()
        assert r1 is r2
        mod._registry_singleton = None  # cleanup


# ---------------------------------------------------------------------------
# Integration tests — real project YAML configs
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
REAL_TEMPLATES_DIR = PROJECT_ROOT / "config" / "templates"


@pytest.mark.integration
class TestRealTemplates:
    def test_real_templates_dir_exists(self) -> None:
        assert REAL_TEMPLATES_DIR.exists(), f"Missing: {REAL_TEMPLATES_DIR}"

    def test_ir12_template_loaded(self) -> None:
        reg = TemplateRegistry(templates_dir=REAL_TEMPLATES_DIR, project_root=PROJECT_ROOT)
        tc = reg.get("DT_BT_14Fr_FO_10E_IR12")
        assert tc.n_steps == 10
        assert tc.is_multi_step is True
        assert len(tc.displacements_mm) == 10

    def test_ir25_template_loaded(self) -> None:
        reg = TemplateRegistry(templates_dir=REAL_TEMPLATES_DIR, project_root=PROJECT_ROOT)
        tc = reg.get("DT_BT_14Fr_FO_10E_IR25")
        assert tc.n_steps == 10
        assert len(tc.displacements_mm) == 10

    def test_sample_catheterization_loaded(self) -> None:
        reg = TemplateRegistry(templates_dir=REAL_TEMPLATES_DIR, project_root=PROJECT_ROOT)
        tc = reg.get("sample_catheterization")
        assert tc.n_steps == 1
        assert tc.is_multi_step is False

    def test_speed_ranges_valid(self) -> None:
        reg = TemplateRegistry(templates_dir=REAL_TEMPLATES_DIR, project_root=PROJECT_ROOT)
        for tc in reg.all_configs():
            assert tc.speed_range.min_mm_s > 0
            assert tc.speed_range.max_mm_s > tc.speed_range.min_mm_s

    def test_displacements_positive(self) -> None:
        reg = TemplateRegistry(templates_dir=REAL_TEMPLATES_DIR, project_root=PROJECT_ROOT)
        for tc in reg.all_configs():
            assert all(d > 0 for d in tc.displacements_mm)

    def test_ir12_speed_range_10_to_25(self) -> None:
        reg = TemplateRegistry(templates_dir=REAL_TEMPLATES_DIR, project_root=PROJECT_ROOT)
        tc = reg.get("DT_BT_14Fr_FO_10E_IR12")
        assert tc.speed_range.min_mm_s == pytest.approx(10.0)
        assert tc.speed_range.max_mm_s == pytest.approx(25.0)
