"""Tests for src/features/splitter.py"""
import numpy as np
import pytest

from src.features.splitter import split_data


class TestSplitData:
    def test_returns_six_arrays(self, small_xy, minimal_cfg):
        X, y = small_xy
        result = split_data(X, y, minimal_cfg)
        assert len(result) == 6

    def test_total_rows_preserved(self, small_xy, minimal_cfg):
        X, y = small_xy
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, minimal_cfg)
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert len(y_train) + len(y_val) + len(y_test) == len(y)

    def test_approximate_ratios(self, minimal_cfg):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1000, 5)).astype(np.float32)
        y = rng.standard_normal(1000).astype(np.float32)
        X_train, X_val, X_test, *_ = split_data(X, y, minimal_cfg)
        assert abs(len(X_train) / 1000 - 0.70) < 0.03
        assert abs(len(X_val)   / 1000 - 0.15) < 0.03
        assert abs(len(X_test)  / 1000 - 0.15) < 0.03

    def test_x_y_sizes_match(self, small_xy, minimal_cfg):
        X, y = small_xy
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, minimal_cfg)
        assert len(X_train) == len(y_train)
        assert len(X_val)   == len(y_val)
        assert len(X_test)  == len(y_test)

    def test_reproducibility(self, small_xy, minimal_cfg):
        X, y = small_xy
        result1 = split_data(X, y, minimal_cfg)
        result2 = split_data(X, y, minimal_cfg)
        for a, b in zip(result1, result2):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_splits(self, minimal_cfg):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 5)).astype(np.float32)
        y = rng.standard_normal(200).astype(np.float32)

        minimal_cfg["split"]["random_seed"] = 0
        r1 = split_data(X, y, minimal_cfg)

        minimal_cfg["split"]["random_seed"] = 99
        r2 = split_data(X, y, minimal_cfg)

        assert not np.array_equal(r1[0], r2[0])  # different train sets

    def test_no_overlap_between_sets(self, small_xy, minimal_cfg):
        X, y = small_xy
        X_train, X_val, X_test, *_ = split_data(X, y, minimal_cfg)

        # Convert rows to sets of tuples to check overlap
        train_set = {tuple(row) for row in X_train}
        val_set   = {tuple(row) for row in X_val}
        test_set  = {tuple(row) for row in X_test}

        assert len(train_set & val_set)  == 0
        assert len(train_set & test_set) == 0
        assert len(val_set   & test_set) == 0
