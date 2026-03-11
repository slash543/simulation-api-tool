"""
Tests for SqlIngestor (ml/sql_ingestor.py).

Strategy
--------
All tests use in-memory SQLite (``sqlite:///:memory:``) — no files written,
no services needed, each test gets a fresh engine.

Coverage
--------
- create_schema() creates all tables
- create_schema() creates ml_training_flat view
- ingest_run() inserts simulation_runs row
- ingest_run() inserts simulation_step_params rows
- ingest_run() inserts pressure_timeseries rows
- mean_speed_mm_s computed correctly
- Idempotent re-ingestion (upsert-by-delete) replaces old data
- query_ml_training_data() returns DataFrame (pandas required)
- query_table() returns table contents
- connection_string property
- Empty step_params ingested without error
- Empty pressure_series ingested without error
- ml_training_flat view returns only COMPLETED runs
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import text

from digital_twin_ui.ml.sql_ingestor import SqlIngestor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IN_MEMORY = "sqlite:///:memory:"


@pytest.fixture()
def db() -> SqlIngestor:
    """Fresh in-memory SQLite database with schema created."""
    ingestor = SqlIngestor(connection_string=IN_MEMORY)
    ingestor.create_schema()
    return ingestor


def _make_step_params(n: int = 3, base_speed: float = 15.0) -> list[dict[str, Any]]:
    return [
        {
            "step_number": i + 1,
            "speed_mm_s": base_speed + i,
            "displacement_mm": 10.0,
            "ramp_duration_s": 10.0 / (base_speed + i),
            "dwell_duration_s": 1.0,
            "t_start_s": float(i * 2),
            "t_end_s": float(i * 2 + 2),
            "time_steps": 20,
        }
        for i in range(n)
    ]


def _make_pressure_series(n: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "time_s": float(i),
            "step_number": (i % 3) + 1,
            "max_pressure_pa": 1000.0 + i * 100,
            "mean_pressure_pa": 800.0 + i * 50,
            "n_faces": 2734,
        }
        for i in range(n)
    ]


def _row_count(db: SqlIngestor, table: str) -> int:
    engine = db._get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
        return result.scalar()


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

class TestCreateSchema:
    def test_tables_created(self) -> None:
        ingestor = SqlIngestor(connection_string=IN_MEMORY)
        ingestor.create_schema()
        engine = ingestor._get_engine()
        inspector = sa.inspect(engine)
        tables = inspector.get_table_names()
        assert "simulation_runs" in tables
        assert "simulation_step_params" in tables
        assert "pressure_timeseries" in tables

    def test_ml_training_flat_view_created(self) -> None:
        ingestor = SqlIngestor(connection_string=IN_MEMORY)
        ingestor.create_schema()
        engine = ingestor._get_engine()
        # Query the view — should not raise
        with engine.connect() as conn:
            conn.execute(text("SELECT * FROM ml_training_flat")).fetchall()

    def test_create_schema_idempotent(self, db: SqlIngestor) -> None:
        # Calling create_schema() twice should not raise
        db.create_schema()
        db.create_schema()


# ---------------------------------------------------------------------------
# ingest_run — basic insertion
# ---------------------------------------------------------------------------

class TestIngestRun:
    def test_simulation_runs_row_inserted(self, db: SqlIngestor) -> None:
        db.ingest_run(
            run_id="run_test_001",
            template_name="DT_BT_14Fr_FO_10E_IR12",
            dwell_time_s=1.0,
            step_params=_make_step_params(3),
            pressure_series=_make_pressure_series(5),
            status="COMPLETED",
            duration_s=42.5,
        )
        assert _row_count(db, "simulation_runs") == 1

    def test_step_params_rows_inserted(self, db: SqlIngestor) -> None:
        db.ingest_run(
            run_id="run_test_002",
            template_name="DT_BT_14Fr_FO_10E_IR12",
            dwell_time_s=1.0,
            step_params=_make_step_params(10),
            pressure_series=_make_pressure_series(5),
            status="COMPLETED",
        )
        assert _row_count(db, "simulation_step_params") == 10

    def test_pressure_series_rows_inserted(self, db: SqlIngestor) -> None:
        db.ingest_run(
            run_id="run_test_003",
            template_name="DT_BT_14Fr_FO_10E_IR12",
            dwell_time_s=1.0,
            step_params=_make_step_params(3),
            pressure_series=_make_pressure_series(7),
            status="COMPLETED",
        )
        assert _row_count(db, "pressure_timeseries") == 7

    def test_mean_speed_computed(self, db: SqlIngestor) -> None:
        step_params = [
            {
                "step_number": 1, "speed_mm_s": 10.0, "displacement_mm": 10.0,
                "ramp_duration_s": 1.0, "dwell_duration_s": 1.0,
                "t_start_s": 0.0, "t_end_s": 2.0, "time_steps": 20,
            },
            {
                "step_number": 2, "speed_mm_s": 20.0, "displacement_mm": 10.0,
                "ramp_duration_s": 0.5, "dwell_duration_s": 1.0,
                "t_start_s": 2.0, "t_end_s": 3.5, "time_steps": 15,
            },
        ]
        db.ingest_run(
            run_id="run_mean_speed",
            template_name="test",
            dwell_time_s=1.0,
            step_params=step_params,
            pressure_series=[],
            status="COMPLETED",
        )
        engine = db._get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT mean_speed_mm_s FROM simulation_runs WHERE run_id='run_mean_speed'")
            ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(15.0)  # (10+20)/2

    def test_optional_fields_none(self, db: SqlIngestor) -> None:
        db.ingest_run(
            run_id="run_none_fields",
            template_name="test",
            dwell_time_s=1.0,
            step_params=[],
            pressure_series=[],
            status="FAILED",
        )
        engine = db._get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT duration_s, mlflow_run_id, mean_speed_mm_s "
                     "FROM simulation_runs WHERE run_id='run_none_fields'")
            ).fetchone()
        assert row[0] is None   # duration_s
        assert row[1] is None   # mlflow_run_id
        assert row[2] is None   # mean_speed_mm_s

    def test_mlflow_run_id_stored(self, db: SqlIngestor) -> None:
        db.ingest_run(
            run_id="run_mlflow",
            template_name="test",
            dwell_time_s=1.0,
            step_params=[],
            pressure_series=[],
            status="COMPLETED",
            mlflow_run_id="mlflow-abc123",
        )
        engine = db._get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT mlflow_run_id FROM simulation_runs WHERE run_id='run_mlflow'")
            ).fetchone()
        assert row[0] == "mlflow-abc123"


# ---------------------------------------------------------------------------
# Idempotent re-ingestion (upsert-by-delete)
# ---------------------------------------------------------------------------

class TestIdempotentIngestion:
    def test_reingest_replaces_run_row(self, db: SqlIngestor) -> None:
        db.ingest_run(
            run_id="run_upsert",
            template_name="original",
            dwell_time_s=1.0,
            step_params=_make_step_params(3),
            pressure_series=[],
            status="COMPLETED",
            duration_s=10.0,
        )
        db.ingest_run(
            run_id="run_upsert",
            template_name="updated",
            dwell_time_s=2.0,
            step_params=_make_step_params(3),
            pressure_series=[],
            status="COMPLETED",
            duration_s=20.0,
        )
        engine = db._get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT template_name, duration_s FROM simulation_runs "
                     "WHERE run_id='run_upsert'")
            ).fetchone()
        assert row[0] == "updated"
        assert row[1] == pytest.approx(20.0)

    def test_reingest_replaces_step_params(self, db: SqlIngestor) -> None:
        db.ingest_run(
            run_id="run_step_upsert",
            template_name="t",
            dwell_time_s=1.0,
            step_params=_make_step_params(5),
            pressure_series=[],
            status="COMPLETED",
        )
        db.ingest_run(
            run_id="run_step_upsert",
            template_name="t",
            dwell_time_s=1.0,
            step_params=_make_step_params(10),
            pressure_series=[],
            status="COMPLETED",
        )
        engine = db._get_engine()
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM simulation_step_params "
                     "WHERE run_id='run_step_upsert'")
            ).scalar()
        assert count == 10

    def test_reingest_replaces_pressure_series(self, db: SqlIngestor) -> None:
        db.ingest_run(
            run_id="run_pres_upsert",
            template_name="t",
            dwell_time_s=1.0,
            step_params=[],
            pressure_series=_make_pressure_series(3),
            status="COMPLETED",
        )
        db.ingest_run(
            run_id="run_pres_upsert",
            template_name="t",
            dwell_time_s=1.0,
            step_params=[],
            pressure_series=_make_pressure_series(8),
            status="COMPLETED",
        )
        engine = db._get_engine()
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM pressure_timeseries "
                     "WHERE run_id='run_pres_upsert'")
            ).scalar()
        assert count == 8

    def test_multiple_runs_independent(self, db: SqlIngestor) -> None:
        for i in range(3):
            db.ingest_run(
                run_id=f"run_{i:03d}",
                template_name="t",
                dwell_time_s=1.0,
                step_params=_make_step_params(2),
                pressure_series=[],
                status="COMPLETED",
            )
        assert _row_count(db, "simulation_runs") == 3
        assert _row_count(db, "simulation_step_params") == 6


# ---------------------------------------------------------------------------
# Empty collections
# ---------------------------------------------------------------------------

class TestEmptyCollections:
    def test_empty_step_params_no_error(self, db: SqlIngestor) -> None:
        db.ingest_run(
            run_id="run_empty_steps",
            template_name="t",
            dwell_time_s=1.0,
            step_params=[],
            pressure_series=[],
            status="FAILED",
        )
        assert _row_count(db, "simulation_runs") == 1
        assert _row_count(db, "simulation_step_params") == 0

    def test_empty_pressure_series_no_error(self, db: SqlIngestor) -> None:
        db.ingest_run(
            run_id="run_empty_pres",
            template_name="t",
            dwell_time_s=1.0,
            step_params=_make_step_params(3),
            pressure_series=[],
            status="COMPLETED",
        )
        assert _row_count(db, "pressure_timeseries") == 0


# ---------------------------------------------------------------------------
# ml_training_flat view
# ---------------------------------------------------------------------------

class TestMlTrainingFlatView:
    def test_view_only_shows_completed(self, db: SqlIngestor) -> None:
        db.ingest_run(
            run_id="run_ok",
            template_name="t",
            dwell_time_s=1.0,
            step_params=_make_step_params(2),
            pressure_series=_make_pressure_series(2),
            status="COMPLETED",
        )
        db.ingest_run(
            run_id="run_fail",
            template_name="t",
            dwell_time_s=1.0,
            step_params=_make_step_params(2),
            pressure_series=[],
            status="FAILED",
        )
        pd = pytest.importorskip("pandas")
        df = db.query_ml_training_data()
        assert len(df) == 1
        assert df.iloc[0]["run_id"] == "run_ok"

    def test_view_speed_step_columns_present(self, db: SqlIngestor) -> None:
        step_params = _make_step_params(10, base_speed=15.0)
        db.ingest_run(
            run_id="run_speed_cols",
            template_name="DT_BT_14Fr_FO_10E_IR12",
            dwell_time_s=1.0,
            step_params=step_params,
            pressure_series=_make_pressure_series(3),
            status="COMPLETED",
        )
        pd = pytest.importorskip("pandas")
        df = db.query_ml_training_data()
        assert len(df) == 1
        for i in range(1, 11):
            assert f"speed_step_{i}" in df.columns


# ---------------------------------------------------------------------------
# query_table
# ---------------------------------------------------------------------------

class TestQueryTable:
    def test_query_simulation_runs(self, db: SqlIngestor) -> None:
        db.ingest_run(
            run_id="run_q1",
            template_name="t",
            dwell_time_s=1.0,
            step_params=[],
            pressure_series=[],
            status="COMPLETED",
        )
        pd = pytest.importorskip("pandas")
        df = db.query_table("simulation_runs")
        assert len(df) == 1
        assert df.iloc[0]["run_id"] == "run_q1"


# ---------------------------------------------------------------------------
# connection_string property
# ---------------------------------------------------------------------------

class TestConnectionString:
    def test_connection_string_property(self) -> None:
        ingestor = SqlIngestor(connection_string=IN_MEMORY)
        assert ingestor.connection_string == IN_MEMORY
