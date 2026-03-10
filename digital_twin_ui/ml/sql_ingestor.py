"""
SQL Ingestor
============
Ingests simulation run data into a relational database for ML training.

Supports SQLite (local dev, no services needed) and PostgreSQL / Azure SQL
(production) via a SQLAlchemy connection string.

Schema
------
Three tables and one view::

    simulation_runs          — one row per simulation
    simulation_step_params   — 10 rows per run (one per insertion step)
    pressure_timeseries      — one row per extracted timestep
    ml_training_flat (VIEW)  — pivoted flat table for ML training

Usage
-----
    from digital_twin_ui.ml.sql_ingestor import SqlIngestor

    # Local SQLite (default)
    db = SqlIngestor()
    db.create_schema()
    db.ingest_run(
        run_id="run_20260310_143000_a1b2",
        template_name="DT_BT_14Fr_FO_10E_IR12",
        dwell_time_s=1.0,
        step_params=[
            {
                "step_number": 1, "speed_mm_s": 15.0, "displacement_mm": 64.0,
                "ramp_duration_s": 4.27, "dwell_duration_s": 1.0,
                "t_start_s": 0.0, "t_end_s": 5.27, "time_steps": 53,
            },
            ...  # 9 more dicts
        ],
        pressure_series=[
            {"time_s": 0.0, "step_number": 1, "max_pressure_pa": 1234.5,
             "mean_pressure_pa": 890.0, "n_faces": 2734},
            ...
        ],
        status="COMPLETED",
        duration_s=42.5,
    )

    df = db.query_ml_training_data()
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy import text

from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_CONNECTION_STRING = "sqlite:///data/synthetic_db.sqlite"


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

_metadata = sa.MetaData()

simulation_runs = sa.Table(
    "simulation_runs",
    _metadata,
    sa.Column("run_id", sa.String(60), primary_key=True),
    sa.Column("template_name", sa.String(120), nullable=False),
    sa.Column("mean_speed_mm_s", sa.Float, nullable=True),
    sa.Column("dwell_time_s", sa.Float, nullable=False),
    sa.Column("status", sa.String(20), nullable=False),
    sa.Column("duration_s", sa.Float, nullable=True),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    ),
    sa.Column("mlflow_run_id", sa.String(100), nullable=True),
)

simulation_step_params = sa.Table(
    "simulation_step_params",
    _metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column(
        "run_id",
        sa.String(60),
        sa.ForeignKey("simulation_runs.run_id"),
        nullable=False,
    ),
    sa.Column("step_number", sa.SmallInteger, nullable=False),
    sa.Column("speed_mm_s", sa.Float, nullable=False),
    sa.Column("displacement_mm", sa.Float, nullable=False),
    sa.Column("ramp_duration_s", sa.Float, nullable=False),
    sa.Column("dwell_duration_s", sa.Float, nullable=False),
    sa.Column("t_start_s", sa.Float, nullable=False),
    sa.Column("t_end_s", sa.Float, nullable=False),
    sa.Column("time_steps", sa.Integer, nullable=False),
)

pressure_timeseries = sa.Table(
    "pressure_timeseries",
    _metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column(
        "run_id",
        sa.String(60),
        sa.ForeignKey("simulation_runs.run_id"),
        nullable=False,
    ),
    sa.Column("time_s", sa.Float, nullable=False),
    sa.Column("step_number", sa.SmallInteger, nullable=True),
    sa.Column("max_pressure_pa", sa.Float, nullable=False),
    sa.Column("mean_pressure_pa", sa.Float, nullable=False),
    sa.Column("n_faces", sa.Integer, nullable=False),
)


# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------


class SqlIngestor:
    """
    Ingests simulation results into a SQL database.

    Supports SQLite (for local development) and PostgreSQL / Azure SQL
    (for production) — just pass the right connection string.

    Args:
        connection_string: SQLAlchemy connection URL.  Defaults to
                           ``sqlite:///data/synthetic_db.sqlite``.
    """

    def __init__(
        self,
        connection_string: str = _DEFAULT_CONNECTION_STRING,
    ) -> None:
        self._connection_string = connection_string
        self._engine: sa.Engine | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def connection_string(self) -> str:
        """The SQLAlchemy connection string."""
        return self._connection_string

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """
        Create all tables if they do not exist.

        Also creates the ``ml_training_flat`` view (or recreates it on
        SQLite where CREATE OR REPLACE VIEW is not supported).
        """
        engine = self._get_engine()
        _metadata.create_all(engine, checkfirst=True)
        self._create_flat_view(engine)
        logger.info(
            "SQL schema ensured on {cs}",
            cs=self._connection_string.split("@")[-1],  # hide credentials
        )

    def _create_flat_view(self, engine: sa.Engine) -> None:
        """Create or replace the ml_training_flat view."""
        is_sqlite = engine.dialect.name == "sqlite"

        if is_sqlite:
            drop_sql = "DROP VIEW IF EXISTS ml_training_flat"
            create_sql = _FLAT_VIEW_SQLITE
        else:
            drop_sql = "DROP VIEW IF EXISTS ml_training_flat"
            create_sql = _FLAT_VIEW_STANDARD

        with engine.begin() as conn:
            conn.execute(text(drop_sql))
            conn.execute(text(create_sql))

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_run(
        self,
        run_id: str,
        template_name: str,
        dwell_time_s: float,
        step_params: list[dict[str, Any]],
        pressure_series: list[dict[str, Any]],
        status: str,
        duration_s: float | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        """
        Ingest a completed simulation run into the database.

        If a run with the same ``run_id`` already exists, it is deleted and
        re-inserted (upsert-by-delete pattern, safe for SQLite and PG).

        Args:
            run_id:          Unique run identifier.
            template_name:   Name of the FEB template used.
            dwell_time_s:    Dwell time used for all steps.
            step_params:     List of per-step dicts (see schema above).
                             Expected keys: step_number, speed_mm_s,
                             displacement_mm, ramp_duration_s, dwell_duration_s,
                             t_start_s, t_end_s, time_steps.
            pressure_series: List of per-timestep dicts.
                             Expected keys: time_s, step_number (optional),
                             max_pressure_pa, mean_pressure_pa, n_faces.
            status:          Run status string (e.g. "COMPLETED").
            duration_s:      Wall-clock duration in seconds.
            mlflow_run_id:   Optional MLflow run ID.
        """
        mean_speed = (
            float(
                sum(p["speed_mm_s"] for p in step_params) / len(step_params)
            )
            if step_params
            else None
        )

        engine = self._get_engine()
        with engine.begin() as conn:
            # Delete existing rows for this run_id (idempotent ingestion)
            conn.execute(
                pressure_timeseries.delete().where(
                    pressure_timeseries.c.run_id == run_id
                )
            )
            conn.execute(
                simulation_step_params.delete().where(
                    simulation_step_params.c.run_id == run_id
                )
            )
            conn.execute(
                simulation_runs.delete().where(
                    simulation_runs.c.run_id == run_id
                )
            )

            # Insert simulation_runs row
            conn.execute(
                simulation_runs.insert().values(
                    run_id=run_id,
                    template_name=template_name,
                    mean_speed_mm_s=mean_speed,
                    dwell_time_s=dwell_time_s,
                    status=status,
                    duration_s=duration_s,
                    created_at=datetime.now(timezone.utc),
                    mlflow_run_id=mlflow_run_id,
                )
            )

            # Insert step params rows
            if step_params:
                conn.execute(
                    simulation_step_params.insert(),
                    [
                        {
                            "run_id": run_id,
                            "step_number": int(p["step_number"]),
                            "speed_mm_s": float(p["speed_mm_s"]),
                            "displacement_mm": float(p["displacement_mm"]),
                            "ramp_duration_s": float(p["ramp_duration_s"]),
                            "dwell_duration_s": float(p["dwell_duration_s"]),
                            "t_start_s": float(p["t_start_s"]),
                            "t_end_s": float(p["t_end_s"]),
                            "time_steps": int(p["time_steps"]),
                        }
                        for p in step_params
                    ],
                )

            # Insert pressure timeseries rows
            if pressure_series:
                conn.execute(
                    pressure_timeseries.insert(),
                    [
                        {
                            "run_id": run_id,
                            "time_s": float(p["time_s"]),
                            "step_number": int(p.get("step_number") or 0) or None,
                            "max_pressure_pa": float(p["max_pressure_pa"]),
                            "mean_pressure_pa": float(p["mean_pressure_pa"]),
                            "n_faces": int(p["n_faces"]),
                        }
                        for p in pressure_series
                    ],
                )

        logger.info(
            "Ingested run '{run_id}' ({n_steps} steps, {n_ts} timesteps)",
            run_id=run_id,
            n_steps=len(step_params),
            n_ts=len(pressure_series),
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_ml_training_data(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        """
        Return the ``ml_training_flat`` view as a pandas DataFrame.

        Each row is one completed simulation run with columns:
            run_id, template_name, dwell_time_s, speed_step_1 … speed_step_10,
            max_pressure_pa, mean_pressure_pa

        Returns:
            pandas.DataFrame

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for query_ml_training_data()"
            ) from exc

        engine = self._get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM ml_training_flat"))
            rows = result.fetchall()
            cols = list(result.keys())

        return pd.DataFrame(rows, columns=cols)

    def query_table(self, table_name: str) -> "pd.DataFrame":  # type: ignore[name-defined]
        """
        Return any table or view as a pandas DataFrame.

        Args:
            table_name: Name of the table or view.

        Returns:
            pandas.DataFrame
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for query_table()") from exc

        engine = self._get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table_name}"))
            rows = result.fetchall()
            cols = list(result.keys())

        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_engine(self) -> sa.Engine:
        """Lazy-create and return the SQLAlchemy engine."""
        if self._engine is None:
            # For SQLite, ensure the parent directory exists
            cs = self._connection_string
            if cs.startswith("sqlite:///") and not cs.startswith("sqlite:////"):
                # relative path sqlite
                db_path = Path(cs[len("sqlite:///"):])
                db_path.parent.mkdir(parents=True, exist_ok=True)
            self._engine = sa.create_engine(cs, echo=False)
        return self._engine


# ---------------------------------------------------------------------------
# VIEW SQL
# ---------------------------------------------------------------------------

# SQLite-compatible version (no PIVOT, uses conditional aggregation)
_FLAT_VIEW_SQLITE = """
CREATE VIEW ml_training_flat AS
SELECT
    r.run_id,
    r.template_name,
    r.dwell_time_s,
    MAX(CASE WHEN sp.step_number = 1  THEN sp.speed_mm_s END) AS speed_step_1,
    MAX(CASE WHEN sp.step_number = 2  THEN sp.speed_mm_s END) AS speed_step_2,
    MAX(CASE WHEN sp.step_number = 3  THEN sp.speed_mm_s END) AS speed_step_3,
    MAX(CASE WHEN sp.step_number = 4  THEN sp.speed_mm_s END) AS speed_step_4,
    MAX(CASE WHEN sp.step_number = 5  THEN sp.speed_mm_s END) AS speed_step_5,
    MAX(CASE WHEN sp.step_number = 6  THEN sp.speed_mm_s END) AS speed_step_6,
    MAX(CASE WHEN sp.step_number = 7  THEN sp.speed_mm_s END) AS speed_step_7,
    MAX(CASE WHEN sp.step_number = 8  THEN sp.speed_mm_s END) AS speed_step_8,
    MAX(CASE WHEN sp.step_number = 9  THEN sp.speed_mm_s END) AS speed_step_9,
    MAX(CASE WHEN sp.step_number = 10 THEN sp.speed_mm_s END) AS speed_step_10,
    pt.max_pressure_pa,
    pt.mean_pressure_pa
FROM simulation_runs r
LEFT JOIN simulation_step_params sp ON sp.run_id = r.run_id
LEFT JOIN (
    SELECT run_id,
           MAX(max_pressure_pa)  AS max_pressure_pa,
           AVG(mean_pressure_pa) AS mean_pressure_pa
    FROM pressure_timeseries
    GROUP BY run_id
) pt ON pt.run_id = r.run_id
WHERE r.status = 'COMPLETED'
GROUP BY r.run_id, r.template_name, r.dwell_time_s,
         pt.max_pressure_pa, pt.mean_pressure_pa
"""

# Standard SQL version (works with PostgreSQL / Azure SQL)
_FLAT_VIEW_STANDARD = """
CREATE VIEW ml_training_flat AS
SELECT
    r.run_id,
    r.template_name,
    r.dwell_time_s,
    MAX(CASE WHEN sp.step_number = 1  THEN sp.speed_mm_s END) AS speed_step_1,
    MAX(CASE WHEN sp.step_number = 2  THEN sp.speed_mm_s END) AS speed_step_2,
    MAX(CASE WHEN sp.step_number = 3  THEN sp.speed_mm_s END) AS speed_step_3,
    MAX(CASE WHEN sp.step_number = 4  THEN sp.speed_mm_s END) AS speed_step_4,
    MAX(CASE WHEN sp.step_number = 5  THEN sp.speed_mm_s END) AS speed_step_5,
    MAX(CASE WHEN sp.step_number = 6  THEN sp.speed_mm_s END) AS speed_step_6,
    MAX(CASE WHEN sp.step_number = 7  THEN sp.speed_mm_s END) AS speed_step_7,
    MAX(CASE WHEN sp.step_number = 8  THEN sp.speed_mm_s END) AS speed_step_8,
    MAX(CASE WHEN sp.step_number = 9  THEN sp.speed_mm_s END) AS speed_step_9,
    MAX(CASE WHEN sp.step_number = 10 THEN sp.speed_mm_s END) AS speed_step_10,
    pt.max_pressure_pa,
    pt.mean_pressure_pa
FROM simulation_runs r
LEFT JOIN simulation_step_params sp ON sp.run_id = r.run_id
LEFT JOIN (
    SELECT run_id,
           MAX(max_pressure_pa)  AS max_pressure_pa,
           AVG(mean_pressure_pa) AS mean_pressure_pa
    FROM pressure_timeseries
    GROUP BY run_id
) pt ON pt.run_id = r.run_id
WHERE r.status = 'COMPLETED'
GROUP BY r.run_id, r.template_name, r.dwell_time_s,
         pt.max_pressure_pa, pt.mean_pressure_pa
"""
