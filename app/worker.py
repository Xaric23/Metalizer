from __future__ import annotations

from app.core.config import get_settings
from app.services.job_store import JobManager, JobStore
from app.services.processor import MetalizerPipeline

settings = get_settings()
pipeline = MetalizerPipeline()
job_store = JobStore(settings.jobs_db_url, retention_hours=settings.job_retention_hours)
job_manager = JobManager(
    pipeline,
    job_store,
    queue_backend=settings.queue_backend,
    redis_url=settings.redis_url,
    redis_queue_name=settings.redis_queue_name,
)


def run_job(job_id: str) -> str | None:
    result = job_manager.run_job(job_id)
    return str(result) if result else None
