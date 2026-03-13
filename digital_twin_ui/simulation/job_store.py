"""
Job Store — SQLite-backed simulation job registry.

Provides a fast, persistent index of all simulation jobs so that
``list_simulation_jobs`` queries the DB in O(log n) regardless of how many
run directories accumulate in runs/.

The DB file lives at ``data/jobs.db`` which is on its own Docker volume
(``datasets_data``) — completely separate from the ``runs/`` volume.

Both the API and Celery worker containers mount ``/app/data``, so both can
read/write the same SQLite file.  WAL mode is enabled so concurrent reads
from the API never block writes from the worker.

Usage
-----
    from digital_twin_ui.simulation.job_store import get_job_store

    store = get_job_store()
    store.insert(run_id=..., task_id=..., design=..., ...)
    store.update_status(run_id, "COMPLETED")
    jobs = store.list_recent(limit=20)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)

_DB_RELATIVE = Path("data/jobs.db")


class JobStore:
    """
    Thin SQLite wrapper for simulation job metadata.

    Args:
        db_path: Absolute path to the ``jobs.db`` file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db = str(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    run_id        TEXT PRIMARY KEY,
                    task_id       TEXT,
                    design        TEXT,
                    configuration TEXT,
                    speeds_json   TEXT,
                    dwell_time_s  REAL,
                    status        TEXT DEFAULT 'PENDING',
                    created_at    TEXT,
                    updated_at    TEXT,
                    run_dir       TEXT,
                    xplt_path     TEXT,
                    error_message TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(
        self,
        run_id: str,
        task_id: str,
        design: str,
        configuration: str,
        speeds_mm_s: list[float],
        dwell_time_s: float,
        run_dir: str,
        xplt_path: str,
    ) -> None:
        """Insert a new job record (PENDING).  Silently ignores duplicates."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO jobs
                    (run_id, task_id, design, configuration, speeds_json,
                     dwell_time_s, status, created_at, updated_at,
                     run_dir, xplt_path)
                VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?)
                """,
                (
                    run_id, task_id, design, configuration,
                    json.dumps(speeds_mm_s), dwell_time_s,
                    now, now, run_dir, xplt_path,
                ),
            )
            conn.commit()

    def update_status(
        self,
        run_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Update the status of an existing job."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status=?, updated_at=?, error_message=?
                WHERE run_id=?
                """,
                (status, now, error_message, run_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_recent(self, limit: int = 20) -> list[dict]:
        """Return the most recent ``limit`` jobs, newest first."""
        limit = min(max(1, limit), 100)
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_by_run_id(self, run_id: str) -> dict | None:
        """Return one job record by run_id, or None if not found."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM jobs WHERE run_id=?", (run_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_by_task_id(self, task_id: str) -> dict | None:
        """Return one job record by Celery task_id, or None if not found."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM jobs WHERE task_id=?", (task_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # One-time migration
    # ------------------------------------------------------------------

    def purge_missing(self) -> int:
        """
        Remove DB records whose run directory no longer exists on disk.

        Safe to call at any time — only deletes rows where the folder is gone.
        Returns the number of records removed.
        """
        with self._connect() as conn:
            cur = conn.execute("SELECT run_id, run_dir FROM jobs")
            rows = cur.fetchall()

        to_delete = [
            row["run_id"]
            for row in rows
            if row["run_dir"] and not Path(row["run_dir"]).exists()
        ]

        if to_delete:
            with self._connect() as conn:
                conn.executemany(
                    "DELETE FROM jobs WHERE run_id=?",
                    [(rid,) for rid in to_delete],
                )
                conn.commit()

        logger.info(
            "purge_missing: removed {n} stale records", n=len(to_delete)
        )
        return len(to_delete)

    def backfill_from_metadata(self, runs_dir: Path) -> int:
        """
        Scan existing ``runs/*/metadata.json`` files and insert any missing
        entries into the DB.

        Called once at API startup so historical runs are visible immediately
        without reprocessing.  Already-indexed runs are skipped (INSERT OR IGNORE).

        Returns the number of newly inserted records.
        """
        if not runs_dir.exists():
            return 0

        n = 0
        for meta_path in sorted(runs_dir.glob("*/metadata.json")):
            try:
                meta: dict = json.loads(meta_path.read_text())
                run_id = meta.get("run_id")
                if not run_id:
                    continue

                # Parse design / configuration out of template_name "design/config"
                template_name: str = meta.get("template_name", "")
                if "/" in template_name:
                    design, configuration = template_name.split("/", 1)
                else:
                    design, configuration = template_name, ""

                # Normalise speeds — old single-step runs store speed_mm_s
                speeds: list[float] = meta.get("speeds_mm_s") or []
                if not speeds:
                    s = meta.get("speed_mm_s")
                    speeds = [float(s)] if s is not None else []

                run_dir = meta.get("run_dir", str(meta_path.parent))
                xplt_path = meta.get(
                    "xplt_file", str(meta_path.parent / "input.xplt")
                )

                # Use started_at as created_at if available
                created_at = meta.get("started_at") or datetime.now(
                    timezone.utc
                ).isoformat()

                # Insert with a dummy created_at that matches metadata
                now = datetime.now(timezone.utc).isoformat()
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO jobs
                            (run_id, task_id, design, configuration, speeds_json,
                             dwell_time_s, status, created_at, updated_at,
                             run_dir, xplt_path)
                        VALUES (?, ?, ?, ?, ?, ?, 'UNKNOWN', ?, ?, ?, ?)
                        """,
                        (
                            run_id, "", design, configuration,
                            json.dumps(speeds), 1.0,
                            created_at, now, run_dir, xplt_path,
                        ),
                    )
                    conn.commit()

                # Update status based on what's actually on disk
                xplt = Path(xplt_path)
                cancel = meta_path.parent / "CANCEL"
                if cancel.exists():
                    self.update_status(run_id, "CANCELLED")
                elif xplt.exists():
                    self.update_status(run_id, "COMPLETED")

                n += 1
            except Exception as exc:
                logger.debug(
                    "backfill: skipping {p}: {e}", p=str(meta_path), e=str(exc)
                )

        logger.info("Job store backfill complete: {n} runs indexed", n=n)
        return n


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: JobStore | None = None


def get_job_store() -> JobStore:
    """Return (and cache) the global JobStore singleton."""
    global _store
    if _store is None:
        from digital_twin_ui.app.core.config import get_settings
        db_path = get_settings().project_root / _DB_RELATIVE
        _store = JobStore(db_path)
    return _store
