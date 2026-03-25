"""
Unit tests for digital_twin_ui.surrogate.csar_engine

Tests cover:
  - build_insertion_depths()
  - load_reference_facets() with various column formats
  - CSAREngine.compute_csar_vs_depth()
  - CSAREngine.compute_from_csv()
  - CSAREngine.csar_to_dict()
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from digital_twin_ui.surrogate.csar_engine import (
    CSAREngine,
    build_insertion_depths,
    load_reference_facets,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_predictor(cp_values: list[float]) -> MagicMock:
    """Return a mock predictor that always predicts the given cp values."""
    pred = MagicMock()
    arr = np.array(cp_values, dtype=np.float32)
    pred.predict_at_depth.return_value = arr
    return pred


@pytest.fixture()
def facets_df():
    """10 facets arranged along Z axis from 0 to 9 mm."""
    n = 10
    return pd.DataFrame(
        {
            "centroid_x": np.zeros(n),
            "centroid_y": np.zeros(n),
            "centroid_z": np.arange(n, dtype=float),
            "facet_area": np.ones(n),
        }
    )


@pytest.fixture()
def facets_csv(facets_df, tmp_path):
    p = tmp_path / "facets.csv"
    facets_df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# build_insertion_depths
# ---------------------------------------------------------------------------

class TestBuildInsertionDepths:
    def test_basic(self):
        d = build_insertion_depths(100.0, 10.0)
        assert d[0] == pytest.approx(0.0)
        assert d[-1] == pytest.approx(100.0)
        assert len(d) == 11

    def test_start_offset(self):
        d = build_insertion_depths(50.0, 10.0, start_mm=10.0)
        assert d[0] == pytest.approx(10.0)
        assert d[-1] == pytest.approx(50.0)

    def test_returns_list_of_floats(self):
        d = build_insertion_depths(30.0, 5.0)
        assert all(isinstance(v, float) for v in d)

    def test_single_point(self):
        d = build_insertion_depths(0.0, 5.0)
        assert len(d) == 1
        assert d[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# load_reference_facets
# ---------------------------------------------------------------------------

class TestLoadReferenceFacets:
    def test_basic_load(self, facets_csv, facets_df):
        loaded = load_reference_facets(facets_csv)
        assert len(loaded) == len(facets_df)
        for col in ["centroid_x", "centroid_y", "centroid_z", "facet_area"]:
            assert col in loaded.columns

    def test_rename_df_facets_columns(self, tmp_path):
        """df_facets uses cx_mm, cy_mm, cz_mm, area_mm2 column names."""
        n = 5
        df = pd.DataFrame(
            {
                "face_id": np.arange(n),
                "cx_mm": np.zeros(n),
                "cy_mm": np.zeros(n),
                "cz_mm": np.arange(n, dtype=float),
                "area_mm2": np.ones(n),
            }
        )
        p = tmp_path / "facets_dfformat.csv"
        df.to_csv(p, index=False)
        loaded = load_reference_facets(p)
        for col in ["centroid_x", "centroid_y", "centroid_z", "facet_area"]:
            assert col in loaded.columns

    def test_drops_insertion_depth(self, tmp_path):
        """insertion_depth column should be dropped (we add it per sample)."""
        df = pd.DataFrame(
            {
                "centroid_x": [0.0],
                "centroid_y": [0.0],
                "centroid_z": [0.0],
                "facet_area": [1.0],
                "insertion_depth": [50.0],
            }
        )
        p = tmp_path / "with_depth.csv"
        df.to_csv(p, index=False)
        loaded = load_reference_facets(p)
        assert "insertion_depth" not in loaded.columns

    def test_deduplication(self, tmp_path):
        """Duplicate rows should be removed."""
        df = pd.DataFrame(
            {
                "centroid_x": [0.0, 0.0],
                "centroid_y": [0.0, 0.0],
                "centroid_z": [0.0, 0.0],
                "facet_area": [1.0, 1.0],
            }
        )
        p = tmp_path / "dup.csv"
        df.to_csv(p, index=False)
        loaded = load_reference_facets(p)
        assert len(loaded) == 1

    def test_missing_columns_raises(self, tmp_path):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_reference_facets(p)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_reference_facets("/nonexistent/facets.csv")


# ---------------------------------------------------------------------------
# CSAREngine
# ---------------------------------------------------------------------------

class TestCSAREngine:
    def test_basic_output_shape(self, facets_df):
        """One depth × two bands → 2 rows."""
        pred = _make_predictor([0.1] * 10)  # all in contact
        engine = CSAREngine(pred)
        depths = [50.0]
        bands = [
            {"zmin": 0, "zmax": 4, "label": "lower"},
            {"zmin": 5, "zmax": 9, "label": "upper"},
        ]
        df = engine.compute_csar_vs_depth(facets_df, depths, bands)
        assert len(df) == 2

    def test_csar_all_in_contact(self, facets_df):
        """All facets in contact → CSAR = 1.0."""
        pred = _make_predictor([1.0] * 10)
        engine = CSAREngine(pred)
        bands = [{"zmin": 0, "zmax": 9, "label": "all"}]
        df = engine.compute_csar_vs_depth(facets_df, [100.0], bands)
        assert df["csar"].iloc[0] == pytest.approx(1.0)

    def test_csar_none_in_contact(self, facets_df):
        """No facets in contact → CSAR = 0.0."""
        pred = _make_predictor([0.0] * 10)
        engine = CSAREngine(pred)
        bands = [{"zmin": 0, "zmax": 9, "label": "all"}]
        df = engine.compute_csar_vs_depth(facets_df, [100.0], bands)
        assert df["csar"].iloc[0] == pytest.approx(0.0)

    def test_csar_half_in_contact(self, facets_df):
        """5 of 10 facets in contact → CSAR ≈ 0.5."""
        cp = [1.0] * 5 + [0.0] * 5  # facets 0–4 in contact, 5–9 not
        pred = _make_predictor(cp)
        engine = CSAREngine(pred)
        bands = [{"zmin": 0, "zmax": 9, "label": "all"}]
        df = engine.compute_csar_vs_depth(facets_df, [100.0], bands)
        assert df["csar"].iloc[0] == pytest.approx(0.5)

    def test_z_band_selection(self, facets_df):
        """Only facets with z in [0,4] should be counted."""
        # All facets in contact, but band only covers z=0..4 (5 facets)
        pred = _make_predictor([1.0] * 10)
        engine = CSAREngine(pred)
        bands = [{"zmin": 0, "zmax": 4, "label": "lower"}]
        df = engine.compute_csar_vs_depth(facets_df, [100.0], bands)
        assert df["n_total_facets"].iloc[0] == 5
        assert df["csar"].iloc[0] == pytest.approx(1.0)

    def test_empty_z_band_gives_nan_csar(self, facets_df):
        """Z band with no facets → NaN CSAR."""
        pred = _make_predictor([1.0] * 10)
        engine = CSAREngine(pred)
        bands = [{"zmin": 100, "zmax": 200, "label": "empty"}]
        df = engine.compute_csar_vs_depth(facets_df, [100.0], bands)
        assert df["n_total_facets"].iloc[0] == 0
        assert np.isnan(df["csar"].iloc[0])

    def test_multiple_depths(self, facets_df):
        """Multiple depths → one row per (depth, band) pair."""
        pred = _make_predictor([0.5] * 10)
        engine = CSAREngine(pred)
        depths = [10.0, 50.0, 100.0]
        bands = [{"zmin": 0, "zmax": 9, "label": "all"}]
        df = engine.compute_csar_vs_depth(facets_df, depths, bands)
        assert len(df) == len(depths)
        assert sorted(df["insertion_depth_mm"].tolist()) == depths

    def test_custom_cp_threshold(self, facets_df):
        """Custom threshold: only facets with cp > 0.5 count as in contact."""
        cp = [0.3] * 5 + [0.8] * 5  # 5 below threshold, 5 above
        pred = _make_predictor(cp)
        engine = CSAREngine(pred, cp_threshold=0.5)
        bands = [{"zmin": 0, "zmax": 9, "label": "all"}]
        df = engine.compute_csar_vs_depth(facets_df, [100.0], bands)
        assert df["n_contact_facets"].iloc[0] == 5

    def test_auto_label(self, facets_df):
        """Band without label should get an auto-generated label."""
        pred = _make_predictor([0.1] * 10)
        engine = CSAREngine(pred)
        bands = [{"zmin": 0, "zmax": 9}]  # no "label" key
        df = engine.compute_csar_vs_depth(facets_df, [100.0], bands)
        assert "z[0,9]" in df["band_label"].iloc[0]

    def test_empty_depths_raises(self, facets_df):
        pred = _make_predictor([0.1] * 10)
        engine = CSAREngine(pred)
        with pytest.raises(ValueError, match="insertion_depths"):
            engine.compute_csar_vs_depth(facets_df, [], [{"zmin": 0, "zmax": 9}])

    def test_empty_bands_raises(self, facets_df):
        pred = _make_predictor([0.1] * 10)
        engine = CSAREngine(pred)
        with pytest.raises(ValueError, match="z_bands"):
            engine.compute_csar_vs_depth(facets_df, [50.0], [])

    def test_missing_geometry_columns_raises(self):
        pred = _make_predictor([0.1])
        engine = CSAREngine(pred)
        bad_df = pd.DataFrame({"x": [1.0]})  # missing required columns
        with pytest.raises(ValueError, match="missing columns"):
            engine.compute_csar_vs_depth(bad_df, [50.0], [{"zmin": 0, "zmax": 1}])

    def test_compute_from_csv(self, facets_csv):
        pred = _make_predictor([1.0] * 10)
        engine = CSAREngine(pred)
        bands = [{"zmin": 0, "zmax": 9, "label": "all"}]
        df = engine.compute_from_csv(facets_csv, [100.0], bands)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_csar_to_dict_keys(self, facets_df):
        pred = _make_predictor([0.5] * 10)
        engine = CSAREngine(pred)
        bands = [{"zmin": 0, "zmax": 9, "label": "all"}]
        result_df = engine.compute_csar_vs_depth(facets_df, [50.0, 100.0], bands)
        d = CSAREngine.csar_to_dict(result_df)
        assert "insertion_depths_mm" in d
        assert "bands" in d
        assert "all" in d["bands"]
        band = d["bands"]["all"]
        assert "csar" in band
        assert "contact_area_mm2" in band
        assert "n_contact_facets" in band

    def test_predictor_called_once_per_depth(self, facets_df):
        """Predictor should be called exactly once per insertion depth."""
        pred = _make_predictor([0.1] * 10)
        engine = CSAREngine(pred)
        depths = [10.0, 50.0, 100.0]
        bands = [{"zmin": 0, "zmax": 9, "label": "all"}]
        engine.compute_csar_vs_depth(facets_df, depths, bands)
        assert pred.predict_at_depth.call_count == len(depths)
