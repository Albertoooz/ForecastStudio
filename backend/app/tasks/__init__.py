"""
Celery app configuration.
"""

from celery import Celery

from app.config import get_settings, sync_langfuse_env_from_settings

settings = get_settings()
sync_langfuse_env_from_settings(settings)

celery_app = Celery(
    "forecaster_tasks",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Route tasks to specific queues
    task_routes={
        "app.tasks.training.*": {"queue": "training"},
        "app.tasks.forecast.*": {"queue": "forecast"},
        "app.tasks.etl.*": {"queue": "etl"},
    },
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["app.tasks"])
