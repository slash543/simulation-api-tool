"""Tests for src/features/engineer.py"""
import numpy as np
import pytest

from src.features.engineer import FeaturePipeline, build_xy


class TestBuildXY:
    def test_shapes(self, sample_df_with_depth, minimal_cfg):
        X, y = build_xy(sample_df_with_depth, minimal_cfg)
        n_features = len(minimal_cfg["features"]["inputs"])
        assert X.shape == (len(sample_df_with_depth), n_features)
        assert y.shape == (len(sample_df_with_depth),)

    def test_dtype_float32(self, sample_df_with_depth, minimal_cfg):
        X, y = build_xy(sample_df_with_depth, minimal_cfg)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_feature_order_matches_config(self, sample_df_with_depth, minimal_cfg):
        X, _ = build_xy(sample_df_with_depth, minimal_cfg)
        for i, col in enumerate(minimal_cfg["features"]["inputs"]):
            # float32 cast introduces sub-1e-6 precision loss vs float64 source
            np.testing.assert_allclose(X[:, i], sample_df_with_depth[col].values, rtol=1e-6)

    def test_target_column(self, sample_df_with_depth, minimal_cfg):
        _, y = build_xy(sample_df_with_depth, minimal_cfg)
        np.testing.assert_allclose(
            y, sample_df_with_depth["contact_pressure"].values, rtol=1e-6
        )

    def test_respects_config_feature_changes(self, sample_df_with_depth, minimal_cfg):
        minimal_cfg["features"]["inputs"] = ["centroid_x", "centroid_y"]
        X, _ = build_xy(sample_df_with_depth, minimal_cfg)
        assert X.shape[1] == 2


class TestFeaturePipeline:
    def test_fit_transform_changes_data(self, small_xy, minimal_cfg):
        X, y = small_xy
        pipeline = FeaturePipeline(minimal_cfg)
        X_s, y_s = pipeline.fit_transform(X, y)
        assert not np.allclose(X, X_s)
        assert not np.allclose(y, y_s)

    def test_standard_scaler_zero_mean(self, small_xy, minimal_cfg):
        X, y = small_xy
        pipeline = FeaturePipeline(minimal_cfg)
        X_s, _ = pipeline.fit_transform(X, y)
        np.testing.assert_allclose(X_s.mean(axis=0), 0.0, atol=1e-5)

    def test_standard_scaler_unit_std(self, small_xy, minimal_cfg):
        X, y = small_xy
        pipeline = FeaturePipeline(minimal_cfg)
        X_s, _ = pipeline.fit_transform(X, y)
        np.testing.assert_allclose(X_s.std(axis=0), 1.0, atol=1e-4)

    def test_minmax_scaler(self, small_xy, minimal_cfg):
        X, y = small_xy
        minimal_cfg["features"]["normalization"]["method"] = "minmax"
        pipeline = FeaturePipeline(minimal_cfg)
        X_s, _ = pipeline.fit_transform(X, y)
        assert X_s.min() >= -1e-6
        assert X_s.max() <= 1.0 + 1e-6

    def test_transform_consistency(self, small_xy, minimal_cfg):
        X, y = small_xy
        pipeline = FeaturePipeline(minimal_cfg)
        pipeline.fit(X, y)
        X_s1, y_s1 = pipeline.transform(X, y)
        X_s2, y_s2 = pipeline.transform(X, y)
        np.testing.assert_array_equal(X_s1, X_s2)
        np.testing.assert_array_equal(y_s1, y_s2)

    def test_inverse_transform_recovers_original(self, small_xy, minimal_cfg):
        X, y = small_xy
        pipeline = FeaturePipeline(minimal_cfg)
        _, y_s = pipeline.fit_transform(X, y)
        y_recovered = pipeline.inverse_transform_y(y_s)
        np.testing.assert_allclose(y_recovered, y, atol=1e-5)

    def test_output_dtype_float32(self, small_xy, minimal_cfg):
        X, y = small_xy
        pipeline = FeaturePipeline(minimal_cfg)
        X_s, y_s = pipeline.fit_transform(X, y)
        assert X_s.dtype == np.float32
        assert y_s.dtype == np.float32

    def test_save_and_load_roundtrip(self, tmp_path, small_xy, minimal_cfg):
        X, y = small_xy
        pipeline = FeaturePipeline(minimal_cfg)
        X_s, y_s = pipeline.fit_transform(X, y)

        pipeline.save(str(tmp_path))
        loaded = FeaturePipeline.load(str(tmp_path), minimal_cfg)

        X_s2, y_s2 = loaded.transform(X, y)
        np.testing.assert_array_equal(X_s, X_s2)
        np.testing.assert_array_equal(y_s, y_s2)

    def test_saved_files_exist(self, tmp_path, small_xy, minimal_cfg):
        X, y = small_xy
        FeaturePipeline(minimal_cfg).fit(X, y).save(str(tmp_path))
        assert (tmp_path / "x_scaler.pkl").exists()
        assert (tmp_path / "y_scaler.pkl").exists()

    def test_fit_transform_same_as_fit_then_transform(self, small_xy, minimal_cfg):
        X, y = small_xy
        p1 = FeaturePipeline(minimal_cfg)
        X_a, y_a = p1.fit_transform(X, y)

        p2 = FeaturePipeline(minimal_cfg)
        p2.fit(X, y)
        X_b, y_b = p2.transform(X, y)

        np.testing.assert_array_equal(X_a, X_b)
        np.testing.assert_array_equal(y_a, y_b)
