"""
Unit tests for digital_twin_ui.surrogate.vtp_processor

Tests cover:
  - VTPData centroids computation
  - VTPData.to_facets_df()
  - VTPProcessor.write() / VTPProcessor.read() round-trip
  - predict_and_save() integration with a mock predictor
  - compute_csar_from_vtp() with a mock predictor
"""
from __future__ import annotations

import base64
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from digital_twin_ui.surrogate.vtp_processor import (
    VTPData,
    VTPProcessor,
    _decode_b64,
    _encode_b64,
    compute_csar_from_vtp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def triangle_vtp():
    """
    A minimal VTPData with 4 points and 2 triangular faces.

    Triangle 0: nodes 0,1,2   (centroid at x=1/3, y=1/3, z=0)
    Triangle 1: nodes 1,2,3   (centroid at x=1,   y=2/3, z=0)
    """
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 1, 0]],
        dtype=np.float32,
    )
    connectivity = np.array([0, 1, 2, 1, 2, 3], dtype=np.int32)
    offsets = np.array([3, 6], dtype=np.int32)
    face_ids = np.array([1, 2], dtype=np.int32)
    areas = np.array([0.5, 0.5], dtype=np.float32)
    cp = np.array([0.1, 0.2], dtype=np.float32)
    return VTPData(
        points=points,
        connectivity=connectivity,
        offsets=offsets,
        face_ids=face_ids,
        areas=areas,
        contact_pressure=cp,
    )


# ---------------------------------------------------------------------------
# Base64 helpers
# ---------------------------------------------------------------------------

class TestBase64Helpers:
    def test_encode_decode_roundtrip(self):
        arr = np.array([1.0, 2.5, -3.0], dtype=np.float32)
        encoded = _encode_b64(arr)
        decoded = _decode_b64(encoded, np.float32)
        np.testing.assert_array_almost_equal(decoded, arr)

    def test_encode_decode_int32(self):
        arr = np.array([0, 1, 100, -5], dtype=np.int32)
        encoded = _encode_b64(arr)
        decoded = _decode_b64(encoded, np.int32)
        np.testing.assert_array_equal(decoded, arr)

    def test_empty_array(self):
        arr = np.array([], dtype=np.float32)
        encoded = _encode_b64(arr)
        decoded = _decode_b64(encoded, np.float32)
        assert len(decoded) == 0

    def test_decode_short_input(self):
        """Decoding < 4 bytes should return empty array, not crash."""
        result = _decode_b64(base64.b64encode(b"ab").decode(), np.float32)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# VTPData
# ---------------------------------------------------------------------------

class TestVTPData:
    def test_n_faces(self, triangle_vtp):
        assert triangle_vtp.n_faces == 2

    def test_n_nodes(self, triangle_vtp):
        assert triangle_vtp.n_nodes == 4

    def test_centroids_shape(self, triangle_vtp):
        c = triangle_vtp.centroids()
        assert c.shape == (2, 3)

    def test_centroids_values(self, triangle_vtp):
        c = triangle_vtp.centroids()
        # Triangle 0: nodes 0(0,0,0), 1(1,0,0), 2(0,1,0) → centroid (1/3, 1/3, 0)
        np.testing.assert_allclose(c[0], [1 / 3, 1 / 3, 0], atol=1e-5)
        # Triangle 1: nodes 1(1,0,0), 2(0,1,0), 3(2,1,0) → centroid (1, 2/3, 0)
        np.testing.assert_allclose(c[1], [1.0, 2 / 3, 0], atol=1e-5)

    def test_to_facets_df(self, triangle_vtp):
        df = triangle_vtp.to_facets_df()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"face_id", "centroid_x", "centroid_y", "centroid_z", "facet_area"}
        assert len(df) == 2
        # Face IDs should match
        assert df["face_id"].tolist() == [1, 2]
        # Areas should match
        np.testing.assert_array_almost_equal(df["facet_area"].values, [0.5, 0.5])


# ---------------------------------------------------------------------------
# VTPProcessor.write / read round-trip
# ---------------------------------------------------------------------------

class TestVTPReadWrite:
    def test_write_creates_file(self, triangle_vtp, tmp_path):
        out = tmp_path / "test.vtp"
        VTPProcessor.write(out, triangle_vtp)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_roundtrip_points(self, triangle_vtp, tmp_path):
        out = tmp_path / "rt.vtp"
        VTPProcessor.write(out, triangle_vtp)
        loaded = VTPProcessor.read(out)
        np.testing.assert_allclose(loaded.points, triangle_vtp.points, atol=1e-5)

    def test_roundtrip_connectivity(self, triangle_vtp, tmp_path):
        out = tmp_path / "rt.vtp"
        VTPProcessor.write(out, triangle_vtp)
        loaded = VTPProcessor.read(out)
        np.testing.assert_array_equal(loaded.connectivity, triangle_vtp.connectivity)

    def test_roundtrip_offsets(self, triangle_vtp, tmp_path):
        out = tmp_path / "rt.vtp"
        VTPProcessor.write(out, triangle_vtp)
        loaded = VTPProcessor.read(out)
        np.testing.assert_array_equal(loaded.offsets, triangle_vtp.offsets)

    def test_roundtrip_face_ids(self, triangle_vtp, tmp_path):
        out = tmp_path / "rt.vtp"
        VTPProcessor.write(out, triangle_vtp)
        loaded = VTPProcessor.read(out)
        np.testing.assert_array_equal(loaded.face_ids, triangle_vtp.face_ids)

    def test_roundtrip_contact_pressure(self, triangle_vtp, tmp_path):
        out = tmp_path / "rt.vtp"
        VTPProcessor.write(out, triangle_vtp)
        loaded = VTPProcessor.read(out)
        np.testing.assert_allclose(loaded.contact_pressure, triangle_vtp.contact_pressure, atol=1e-5)

    def test_write_custom_pressure(self, triangle_vtp, tmp_path):
        new_cp = np.array([0.99, 1.23], dtype=np.float32)
        out = tmp_path / "custom.vtp"
        VTPProcessor.write(out, triangle_vtp, contact_pressure=new_cp)
        loaded = VTPProcessor.read(out)
        np.testing.assert_allclose(loaded.contact_pressure, new_cp, atol=1e-5)

    def test_write_creates_parent_dirs(self, triangle_vtp, tmp_path):
        out = tmp_path / "a" / "b" / "test.vtp"
        VTPProcessor.write(out, triangle_vtp)
        assert out.exists()

    def test_read_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VTPProcessor.read(tmp_path / "nonexistent.vtp")

    def test_write_wrong_pressure_length_raises(self, triangle_vtp, tmp_path):
        with pytest.raises(ValueError, match="length"):
            VTPProcessor.write(
                tmp_path / "bad.vtp",
                triangle_vtp,
                contact_pressure=np.array([0.1]),  # wrong length
            )


# ---------------------------------------------------------------------------
# predict_and_save
# ---------------------------------------------------------------------------

class TestPredictAndSave:
    def _make_predictor(self, return_cp):
        pred = MagicMock()
        pred.predict_at_depth.return_value = np.array(return_cp, dtype=np.float32)
        return pred

    def test_predict_and_save_creates_file(self, triangle_vtp, tmp_path):
        pred = self._make_predictor([0.5, 0.6])
        out = tmp_path / "pred.vtp"
        written = VTPProcessor.predict_and_save(triangle_vtp, pred, 100.0, out)
        assert Path(written).exists()

    def test_predict_and_save_calls_predictor(self, triangle_vtp, tmp_path):
        pred = self._make_predictor([0.5, 0.6])
        VTPProcessor.predict_and_save(triangle_vtp, pred, 150.0, tmp_path / "p.vtp")
        pred.predict_at_depth.assert_called_once()
        _, call_depth = pred.predict_at_depth.call_args[0]
        assert call_depth == 150.0

    def test_predict_and_save_writes_correct_pressure(self, triangle_vtp, tmp_path):
        expected_cp = [0.33, 0.77]
        pred = self._make_predictor(expected_cp)
        out = tmp_path / "pred.vtp"
        VTPProcessor.predict_and_save(triangle_vtp, pred, 50.0, out)
        loaded = VTPProcessor.read(out)
        np.testing.assert_allclose(loaded.contact_pressure, expected_cp, atol=1e-5)


# ---------------------------------------------------------------------------
# compute_csar_from_vtp
# ---------------------------------------------------------------------------

class TestComputeCSARFromVTP:
    def _make_predictor(self, cp_values):
        """Return a mock predictor whose predict_at_depth returns cp_values."""
        pred = MagicMock()
        pred.predict_at_depth.return_value = np.array(cp_values, dtype=np.float32)
        return pred

    def test_returns_dataframe(self, triangle_vtp):
        pred = self._make_predictor([0.1, 0.0])  # face 0 in contact, face 1 not
        depths = [50.0, 100.0]
        z_bands = [{"zmin": -10, "zmax": 10, "label": "all"}]
        df = compute_csar_from_vtp(triangle_vtp, pred, depths, z_bands)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_csar_range(self, triangle_vtp):
        """CSAR must be in [0, 1]."""
        pred = self._make_predictor([0.1, 0.1])  # both in contact
        depths = [100.0]
        z_bands = [{"zmin": -100, "zmax": 100, "label": "all"}]
        df = compute_csar_from_vtp(triangle_vtp, pred, depths, z_bands)
        csar_vals = df["csar"].dropna().values
        assert all(0 <= v <= 1 for v in csar_vals)

    def test_empty_band_gives_nan(self, triangle_vtp):
        """Z band with no facets should yield NaN CSAR."""
        pred = self._make_predictor([0.1, 0.1])
        depths = [50.0]
        z_bands = [{"zmin": 999, "zmax": 1000, "label": "empty"}]
        df = compute_csar_from_vtp(triangle_vtp, pred, depths, z_bands)
        assert df["n_total_facets"].iloc[0] == 0
