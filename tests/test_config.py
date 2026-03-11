"""
Tests for the configuration system.

Coverage:
- _find_project_root()
- _apply_env_overrides()
- load_settings() — defaults, YAML loading, env overrides, invalid YAML
- get_settings() — singleton caching
- Settings derived path properties
- SimulationConfig — all fields and physics invariants
- DOEConfig, MLflowConfig, MLConfig, APIConfig, CeleryConfig, LoggingConfig
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from digital_twin_ui.app.core.config import (
    Settings,
    SimulationConfig,
    DOEConfig,
    MLflowConfig,
    MLConfig,
    APIConfig,
    CeleryConfig,
    LoggingConfig,
    _apply_env_overrides,
    _find_project_root,
    get_settings,
    load_settings,
)
from tests.conftest import patch_env


# ---------------------------------------------------------------------------
# _find_project_root
# ---------------------------------------------------------------------------

class TestFindProjectRoot:
    def test_returns_path_object(self):
        assert isinstance(_find_project_root(), Path)

    def test_finds_root_by_pyproject_or_config(self):
        root = _find_project_root()
        has_pyproject = (root / "pyproject.toml").exists()
        has_config = (root / "config").is_dir()
        assert has_pyproject or has_config

    def test_root_is_absolute(self):
        assert _find_project_root().is_absolute()


# ---------------------------------------------------------------------------
# _apply_env_overrides
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    def test_overrides_string_field(self):
        raw = {"simulation": {"simulator_executable": "febio4"}}
        with patch_env({"DTUI__SIMULATION__SIMULATOR_EXECUTABLE": "febio4-avx2"}):
            result = _apply_env_overrides(raw)
        assert result["simulation"]["simulator_executable"] == "febio4-avx2"

    def test_overrides_numeric_field(self):
        raw = {"api": {"port": "8000"}}
        with patch_env({"DTUI__API__PORT": "9000"}):
            result = _apply_env_overrides(raw)
        assert result["api"]["port"] == "9000"

    def test_overrides_multiple_fields(self):
        raw = {"api": {"port": "8000", "host": "0.0.0.0"}}
        with patch_env({"DTUI__API__PORT": "9000", "DTUI__API__HOST": "127.0.0.1"}):
            result = _apply_env_overrides(raw)
        assert result["api"]["port"] == "9000"
        assert result["api"]["host"] == "127.0.0.1"

    def test_ignores_non_prefixed_vars(self):
        raw = {"simulation": {"simulator_executable": "febio4"}}
        with patch_env({"SOME_OTHER_VAR": "value"}):
            result = _apply_env_overrides(raw)
        assert result["simulation"]["simulator_executable"] == "febio4"

    def test_ignores_malformed_one_part(self):
        raw = {"simulation": {}}
        with patch_env({"DTUI__ONLYONE": "value"}):
            result = _apply_env_overrides(raw)
        assert result == {"simulation": {}}

    def test_ignores_unknown_section(self):
        """Env override for a section not present in raw dict is silently skipped."""
        raw = {"api": {"port": "8000"}}
        with patch_env({"DTUI__UNKNOWN_SECTION__FIELD": "val"}):
            result = _apply_env_overrides(raw)
        assert "unknown_section" not in result

    def test_returns_same_dict_reference(self):
        """_apply_env_overrides mutates and returns the original dict."""
        raw = {"api": {"port": "8000"}}
        result = _apply_env_overrides(raw)
        assert result is raw


# ---------------------------------------------------------------------------
# load_settings
# ---------------------------------------------------------------------------

class TestLoadSettings:
    def test_returns_settings_instance(self, tmp_path):
        s = load_settings(config_path=tmp_path / "nonexistent.yaml")
        assert isinstance(s, Settings)

    def test_defaults_when_no_yaml(self, tmp_path):
        s = load_settings(config_path=tmp_path / "nonexistent.yaml")
        assert s.simulation.displacement_mm == 10.0
        assert s.simulation.num_steps == 2
        assert s.simulation.simulator_executable == "febio4"
        assert s.simulation.simulator_args == ["-i"]
        assert s.doe.speed_min_mm_s == 10.0
        assert s.doe.speed_max_mm_s == 25.0
        assert s.ml.hidden_dims == [64, 128, 256]

    def test_loads_custom_yaml(self, tmp_path):
        cfg = tmp_path / "simulation.yaml"
        cfg.write_text(
            "simulation:\n"
            "  simulator_executable: custom_sim\n"
            "  displacement_mm: 20.0\n"
        )
        s = load_settings(config_path=cfg)
        assert s.simulation.simulator_executable == "custom_sim"
        assert s.simulation.displacement_mm == 20.0

    def test_yaml_partial_override_keeps_defaults(self, tmp_path):
        """YAML overriding one field should not reset other fields to None."""
        cfg = tmp_path / "simulation.yaml"
        cfg.write_text("simulation:\n  displacement_mm: 15.0\n")
        s = load_settings(config_path=cfg)
        assert s.simulation.displacement_mm == 15.0
        assert s.simulation.simulator_executable == "febio4"  # default preserved

    def test_empty_yaml_uses_defaults(self, tmp_path):
        cfg = tmp_path / "simulation.yaml"
        cfg.write_text("")
        s = load_settings(config_path=cfg)
        assert isinstance(s, Settings)
        assert s.simulation.displacement_mm == 10.0

    def test_env_override_applied_on_load(self, tmp_path, clean_settings_cache):
        cfg = tmp_path / "simulation.yaml"
        cfg.write_text("api:\n  port: 8000\n")
        with patch_env({"DTUI__API__PORT": "9999"}):
            s = load_settings(config_path=cfg)
        # env overrides are string-level; Pydantic coerces to int
        assert str(s.api.port) == "9999"


# ---------------------------------------------------------------------------
# get_settings — singleton caching
# ---------------------------------------------------------------------------

class TestGetSettingsSingleton:
    def test_returns_same_object_twice(self, clean_settings_cache):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clears_correctly(self, clean_settings_cache):
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # Different object after cache clear, but same content
        assert s1 is not s2
        assert s1.simulation.displacement_mm == s2.simulation.displacement_mm


# ---------------------------------------------------------------------------
# Derived path properties
# ---------------------------------------------------------------------------

class TestDerivedPaths:
    def test_base_feb_path_abs_is_absolute(self, tmp_path):
        s = load_settings(config_path=tmp_path / "nonexistent.yaml")
        assert s.base_feb_path_abs.is_absolute()

    def test_runs_dir_abs_is_absolute(self, tmp_path):
        s = load_settings(config_path=tmp_path / "nonexistent.yaml")
        assert s.runs_dir_abs.is_absolute()

    def test_log_dir_abs_is_absolute(self, tmp_path):
        s = load_settings(config_path=tmp_path / "nonexistent.yaml")
        assert s.log_dir_abs.is_absolute()

    def test_dataset_path_abs_is_absolute(self, tmp_path):
        s = load_settings(config_path=tmp_path / "nonexistent.yaml")
        assert s.dataset_path_abs.is_absolute()

    def test_checkpoint_dir_abs_is_absolute(self, tmp_path):
        s = load_settings(config_path=tmp_path / "nonexistent.yaml")
        assert s.checkpoint_dir_abs.is_absolute()

    def test_mlflow_tracking_uri_abs_local(self, tmp_path):
        s = load_settings(config_path=tmp_path / "nonexistent.yaml")
        uri = s.mlflow_tracking_uri_abs
        # Default is a relative path → should be made absolute string
        assert uri.startswith("/") or uri.startswith("file://")

    def test_mlflow_tracking_uri_http_unchanged(self, tmp_path):
        cfg = tmp_path / "sim.yaml"
        cfg.write_text("mlflow:\n  tracking_uri: 'http://localhost:5000'\n")
        s = load_settings(config_path=cfg)
        assert s.mlflow_tracking_uri_abs == "http://localhost:5000"

    def test_absolute_base_feb_path_not_changed(self, tmp_path):
        abs_path = str(tmp_path / "custom.feb")
        cfg = tmp_path / "sim.yaml"
        cfg.write_text(f"simulation:\n  base_feb_path: '{abs_path}'\n")
        s = load_settings(config_path=cfg)
        assert str(s.base_feb_path_abs) == abs_path


# ---------------------------------------------------------------------------
# Sub-model validation
# ---------------------------------------------------------------------------

class TestSimulationConfig:
    def test_default_executable(self):
        c = SimulationConfig()
        assert c.simulator_executable == "febio4"

    def test_default_args(self):
        c = SimulationConfig()
        assert c.simulator_args == ["-i"]

    def test_run_command_assembly(self):
        """Verify executable + args + path forms the correct command."""
        c = SimulationConfig()
        input_path = Path("/runs/run_001/input.feb")
        cmd = [c.simulator_executable, *c.simulator_args, str(input_path)]
        assert cmd == ["febio4", "-i", "/runs/run_001/input.feb"]

    def test_physics_default_speed(self):
        """Default config encodes a 5 mm/s insertion speed."""
        c = SimulationConfig()
        default_end_time = 4.0  # from template LC1 end point
        duration = default_end_time - c.loadcurve_start_time
        speed = c.displacement_mm / duration
        assert speed == pytest.approx(5.0)

    def test_speed_formula_range_min(self):
        """At speed=4, insertion duration = 2.5s."""
        c = SimulationConfig()
        duration = c.displacement_mm / 4.0
        assert duration == pytest.approx(2.5)

    def test_speed_formula_range_max(self):
        """At speed=6, insertion duration ≈ 1.667s."""
        c = SimulationConfig()
        duration = c.displacement_mm / 6.0
        assert duration == pytest.approx(10.0 / 6.0)

    def test_step_size_positive(self):
        assert SimulationConfig().default_step_size > 0

    def test_insertion_step_id(self):
        assert SimulationConfig().insertion_step_id == 2

    def test_loadcurve_start_equals_step1_duration(self):
        """LC start time must equal the duration of Step 1."""
        c = SimulationConfig()
        assert c.loadcurve_start_time == pytest.approx(c.step1_duration_s)


class TestDOEConfig:
    def test_default_range(self):
        c = DOEConfig()
        assert c.speed_min_mm_s == pytest.approx(10.0)
        assert c.speed_max_mm_s == pytest.approx(25.0)

    def test_range_is_valid(self):
        c = DOEConfig()
        assert c.speed_min_mm_s < c.speed_max_mm_s

    def test_default_sampler(self):
        assert DOEConfig().default_sampler == "lhs"

    def test_default_num_samples_positive(self):
        assert DOEConfig().default_num_samples > 0


class TestMLflowConfig:
    def test_default_experiment_name(self):
        assert MLflowConfig().experiment_name == "catheter_insertion"

    def test_default_tracking_uri_set(self):
        assert MLflowConfig().tracking_uri != ""


class TestMLConfig:
    def test_hidden_dims(self):
        assert MLConfig().hidden_dims == [64, 128, 256]

    def test_val_fraction_in_range(self):
        c = MLConfig()
        assert 0.0 < c.val_fraction < 1.0

    def test_patience_positive(self):
        assert MLConfig().patience > 0

    def test_learning_rate_positive(self):
        assert MLConfig().learning_rate > 0

    def test_batch_size_positive(self):
        assert MLConfig().batch_size > 0


class TestAPIConfig:
    def test_default_port(self):
        assert APIConfig().port == 8000

    def test_default_host(self):
        assert APIConfig().host == "0.0.0.0"


class TestCeleryConfig:
    def test_broker_url_set(self):
        assert "redis" in CeleryConfig().broker_url

    def test_result_backend_set(self):
        assert "redis" in CeleryConfig().result_backend


class TestLoggingConfig:
    def test_default_level(self):
        assert LoggingConfig().level in {"DEBUG", "INFO", "WARNING", "ERROR"}

    def test_log_dir_is_path(self):
        assert isinstance(LoggingConfig().log_dir, Path)
