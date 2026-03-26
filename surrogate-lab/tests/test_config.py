"""Tests for src/utils/config.py"""
import pytest
import yaml

from src.utils.config import get_feature_names, get_target_name, load_config


class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path):
        cfg_data = {"features": {"inputs": ["a", "b"], "target": "y"}}
        f = tmp_path / "config.yaml"
        f.write_text(yaml.dump(cfg_data))
        result = load_config(str(f))
        assert result == cfg_data

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_returns_dict(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text("key: value\n")
        assert isinstance(load_config(str(f)), dict)


class TestGetFeatureNames:
    def test_returns_list(self, minimal_cfg):
        result = get_feature_names(minimal_cfg)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_correct_values(self, minimal_cfg):
        result = get_feature_names(minimal_cfg)
        assert "centroid_x" in result
        assert "insertion_depth" in result

    def test_reflects_config_changes(self, minimal_cfg):
        minimal_cfg["features"]["inputs"] = ["a", "b", "c"]
        assert get_feature_names(minimal_cfg) == ["a", "b", "c"]


class TestGetTargetName:
    def test_returns_string(self, minimal_cfg):
        assert isinstance(get_target_name(minimal_cfg), str)

    def test_correct_value(self, minimal_cfg):
        assert get_target_name(minimal_cfg) == "contact_pressure"

    def test_reflects_config_changes(self, minimal_cfg):
        minimal_cfg["features"]["target"] = "force"
        assert get_target_name(minimal_cfg) == "force"
