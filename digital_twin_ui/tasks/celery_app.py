"""
Celery application factory for the Digital Twin UI platform.

The application is configured from ``config/simulation.yaml``
(``celery.broker_url`` and ``celery.result_backend``).  All values can be
overridden with environment variables::

    DTUI__CELERY__BROKER_URL=redis://myredis:6379/0
    DTUI__CELERY__RESULT_BACKEND=redis://myredis:6379/0

Usage (production)::

    # Start worker
    celery -A digital_twin_ui.tasks.celery_app worker --loglevel=info

    # Start beat scheduler (if periodic tasks are registered)
    celery -A digital_twin_ui.tasks.celery_app beat --loglevel=info

Usage (application code)::

    from digital_twin_ui.tasks.celery_app import celery_app
"""

from __future__ import annotations

from celery import Celery

from digital_twin_ui.app.core.config import get_settings


def create_celery_app() -> Celery:
    """
    Construct and configure the Celery application.

    Returns:
        Configured :class:`celery.Celery` instance.
    """
    cfg = get_settings()

    app = Celery(
        "digital_twin_ui",
        broker=cfg.celery.broker_url,
        backend=cfg.celery.result_backend,
        include=["digital_twin_ui.tasks.simulation_tasks"],
    )

    app.conf.update(
        # Serialisation
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # Timezone
        timezone="UTC",
        enable_utc=True,
        # Result expiry (24 h)
        result_expires=86400,
        # Retry defaults — tasks opt-in individually
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        # Prevent duplicate result storage
        task_ignore_result=False,
        # Worker concurrency — default to 1 (FEBio is single-process)
        worker_concurrency=1,
    )

    return app


#: Module-level singleton — import and use throughout the application.
celery_app: Celery = create_celery_app()
