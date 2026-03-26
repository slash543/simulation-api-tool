"""Tests for src/data/schema.py"""
import pandas as pd
import pytest

from src.data.schema import validate


class TestValidate:
    def test_passes_with_all_required_columns(self, sample_df_with_depth, minimal_cfg):
        validate(sample_df_with_depth, minimal_cfg)  # should not raise

    def test_raises_on_missing_feature_column(self, sample_df_with_depth, minimal_cfg):
        df = sample_df_with_depth.drop(columns=["centroid_x"])
        with pytest.raises(ValueError, match="centroid_x"):
            validate(df, minimal_cfg)

    def test_raises_on_missing_target_column(self, sample_df_with_depth, minimal_cfg):
        df = sample_df_with_depth.drop(columns=["contact_pressure"])
        with pytest.raises(ValueError, match="contact_pressure"):
            validate(df, minimal_cfg)

    def test_raises_on_multiple_missing_columns(self, minimal_cfg):
        df = pd.DataFrame({"centroid_x": [1.0]})
        with pytest.raises(ValueError):
            validate(df, minimal_cfg)

    def test_extra_columns_are_allowed(self, sample_df_with_depth, minimal_cfg):
        df = sample_df_with_depth.copy()
        df["extra_col"] = 0
        validate(df, minimal_cfg)  # should not raise

    def test_target_changed_via_config(self, sample_df_with_depth, minimal_cfg):
        minimal_cfg["features"]["target"] = "nonexistent_target"
        with pytest.raises(ValueError, match="nonexistent_target"):
            validate(sample_df_with_depth, minimal_cfg)
