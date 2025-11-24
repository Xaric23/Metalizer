from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from loguru import logger
from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, create_engine, delete, insert, select, update
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import sessionmaker


JobStatus = Literal["queued", "processing", "complete", "error"]


@dataclass
class JobRecord:
    job_id: str
    input_path: Path
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    output_path: Path | None = None
    error: str | None = None


metadata = MetaData()

jobs_table = Table(
    "jobs",
    metadata,
    Column("job_id", String(64), primary_key=True),
    Column("input_path", Text, nullable=False),
    Column("output_path", Text, nullable=True),
    Column("status", String(32), nullable=False),
    Column("error", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


class JobStore:
    def __init__(self, database_url: str | URL, retention_hours: int = 24) -> None:
        self.database_url = str(database_url)
        self.retention_hours = retention_hours
        self.engine: Engine = create_engine(
            self.database_url,
            future=True,
            pool_pre_ping=True,
        )
        metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def create_job(self, input_path: Path) -> JobRecord:
        job_id = uuid4().hex
        now = datetime.now(timezone.utc)
        with self.session_factory() as session:
            session.execute(
                insert(jobs_table).values(
                    job_id=job_id,
                    input_path=str(input_path),
                    status="queued",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()
        return JobRecord(job_id=job_id, input_path=input_path, status="queued", created_at=now, updated_at=now)

    def update_status(
        self,
        job_id: str,
        *,
        status: JobStatus,
        output_path: Path | None = None,
        error: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        with self.session_factory() as session:
            session.execute(
                update(jobs_table)
                .where(jobs_table.c.job_id == job_id)
                .values(
                    status=status,
                    output_path=str(output_path) if output_path else None,
                    error=error,
                    updated_at=now,
                )
            )
            session.commit()

    def get(self, job_id: str) -> JobRecord:
        with self.session_factory() as session:
            row = session.execute(
                select(jobs_table).where(jobs_table.c.job_id == job_id)
            ).mappings().first()
        if not row:
            raise KeyError(job_id)
        return self._row_to_record(row)

    def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        with self.session_factory() as session:
            rows = (
                session.execute(
                    select(jobs_table).order_by(jobs_table.c.created_at.desc()).limit(limit)
                )
                .mappings()
                .all()
            )
        return [self._row_to_record(row) for row in rows]

    def delete_job(self, job_id: str, delete_files: bool = True) -> None:
        try:
            record = self.get(job_id)
        except KeyError:
            return
        if delete_files:
            self._safe_unlink(record.input_path)
            if record.output_path:
                self._safe_unlink(record.output_path)
        with self.session_factory() as session:
            session.execute(delete(jobs_table).where(jobs_table.c.job_id == job_id))
            session.commit()

    def cleanup_obsolete(self) -> None:
        if self.retention_hours <= 0:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        with self.session_factory() as session:
            rows = (
                session.execute(
                    select(jobs_table.c.job_id)
                    .where(jobs_table.c.status.in_(["complete", "error"]))
                    .where(jobs_table.c.updated_at <= cutoff)
                )
                .scalars()
                .all()
            )
        for job_id in rows:
            logger.debug("Removing expired job", job_id=job_id)
            self.delete_job(job_id)

    def _row_to_record(self, row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            input_path=Path(row["input_path"]),
            status=row["status"],  # type: ignore[assignment]
            output_path=Path(row["output_path"]) if row["output_path"] else None,
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _safe_unlink(self, path: Path | None) -> None:
        if path and path.exists():
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Failed to delete file", path=str(path), error=str(exc))


class JobManager:
    def __init__(self, pipeline, store: JobStore, *, queue_backend: str = "local", redis_url: str | None = None, redis_queue_name: str = "metalizer") -> None:
        self.pipeline = pipeline
        self.store = store
        self.queue_backend = queue_backend
        self.redis_url = redis_url
        self.redis_queue_name = redis_queue_name
        self._queue = None
        if self.queue_backend == "rq":
            self._queue = self._build_rq_queue()

    def _build_rq_queue(self):
        import redis
        from rq import Queue

        connection = redis.from_url(self.redis_url or "redis://localhost:6379/0")
        return Queue(self.redis_queue_name, connection=connection)

    def enqueue(self, job_id: str, background_tasks=None) -> None:
        if self.queue_backend == "rq" and self._queue is not None:
            self._queue.enqueue("app.worker.run_job", job_id)
            return
        if background_tasks is None:
            self.run_job(job_id)
            return
        background_tasks.add_task(self.run_job, job_id)

    def run_job(self, job_id: str) -> Path | None:
        try:
            self.store.update_status(job_id, status="processing")
            record = self.store.get(job_id)
        except KeyError:
            logger.error("Job not found", job_id=job_id)
            return None

        try:
            output_path = self.pipeline.process(record.input_path)
        except Exception as exc:  # pragma: no cover - pipeline errors covered elsewhere
            logger.exception("Job execution failed", job_id=job_id)
            self.store.update_status(job_id, status="error", error=str(exc))
            raise
        else:
            self.store.update_status(job_id, status="complete", output_path=output_path)
            self.store.cleanup_obsolete()
            return output_path