"""Tests for src/data/loader.py"""
import numpy as np
import pandas as pd
import pytest

from src.data.loader import _add_insertion_depth, load_simulation_data


class TestAddInsertionDepth:
    def test_linear_conversion(self, sample_df, minimal_cfg):
        result = _add_insertion_depth(sample_df, minimal_cfg)
        expected = sample_df["time_step"] * minimal_cfg["data"]["time_to_depth"]["scale"]
        np.testing.assert_array_almost_equal(result["insertion_depth"].values, expected.values)

    def test_skips_when_already_present(self, sample_df, minimal_cfg):
        df = sample_df.copy()
        df["insertion_depth"] = 99.0
        result = _add_insertion_depth(df, minimal_cfg)
        assert (result["insertion_depth"] == 99.0).all()

    def test_warns_when_neither_column_present(self, minimal_cfg, caplog):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _add_insertion_depth(df, minimal_cfg)
        assert "insertion_depth" not in result.columns

    def test_does_not_modify_original(self, sample_df, minimal_cfg):
        original_cols = set(sample_df.columns)
        _add_insertion_depth(sample_df, minimal_cfg)
        assert set(sample_df.columns) == original_cols  # original unchanged

    def test_unknown_method_raises(self, sample_df, minimal_cfg):
        minimal_cfg["data"]["time_to_depth"]["method"] = "quadratic"
        with pytest.raises(ValueError, match="quadratic"):
            _add_insertion_depth(sample_df, minimal_cfg)


class TestLoadSimulationData:
    def _write_csv(self, tmp_path, df, name="sim.csv"):
        path = tmp_path / name
        df.to_csv(path, index=False)
        return path

    def test_loads_single_file(self, tmp_path, sample_df, minimal_cfg):
        path = self._write_csv(tmp_path, sample_df)
        result = load_simulation_data(minimal_cfg, path=str(path))
        assert len(result) == len(sample_df)
        assert "insertion_depth" in result.columns

    def test_loads_multiple_files_from_directory(self, tmp_path, sample_df, minimal_cfg):
        self._write_csv(tmp_path, sample_df, "a.csv")
        self._write_csv(tmp_path, sample_df, "b.csv")
        minimal_cfg["data"]["source"] = str(tmp_path)
        result = load_simulation_data(minimal_cfg)
        assert len(result) == 2 * len(sample_df)

    def test_raises_when_no_files_found(self, tmp_path, minimal_cfg):
        minimal_cfg["data"]["source"] = str(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_simulation_data(minimal_cfg)

    def test_raises_on_missing_required_column(self, tmp_path, sample_df, minimal_cfg):
        df = sample_df.drop(columns=["contact_pressure"])
        path = self._write_csv(tmp_path, df)
        with pytest.raises(ValueError):
            load_simulation_data(minimal_cfg, path=str(path))

    def test_concatenates_correctly(self, tmp_path, sample_df, minimal_cfg):
        self._write_csv(tmp_path, sample_df.iloc[:100], "part1.csv")
        self._write_csv(tmp_path, sample_df.iloc[100:], "part2.csv")
        minimal_cfg["data"]["source"] = str(tmp_path)
        result = load_simulation_data(minimal_cfg)
        assert len(result) == len(sample_df)
        assert result.index.tolist() == list(range(len(sample_df)))  # reset index
