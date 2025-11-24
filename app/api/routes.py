from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from app.api.dependencies import require_assets_api_key
from app.core.config import get_settings
from app.services.assets import list_assets, save_asset
from app.services.job_store import JobManager, JobRecord, JobStore
from app.services.processor import MetalizerPipeline
from app.utils.audio_io import async_persist_upload


router = APIRouter()
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

ACCEPTED_MIME = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/x-m4a"}
ASSET_ALLOWED_MIME = {"audio/wav", "audio/x-wav", "audio/mpeg"}


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/remix")
async def remix_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    background: bool = Query(False),
) -> Response:
    if file.content_type not in ACCEPTED_MIME:
        raise HTTPException(status_code=415, detail="Unsupported audio format")

    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    sanitized_name = Path(file.filename or "upload").stem[:50]
    unique_name = f"{sanitized_name}_{uuid4().hex}{suffix}"
    persisted_input = settings.input_dir / unique_name

    logger.info(
        "Received upload",
        filename=file.filename,
        content_type=file.content_type,
        destination=str(persisted_input),
    )

    await async_persist_upload(file, persisted_input)

    job = job_store.create_job(persisted_input)

    if background:
        job_manager.enqueue(job.job_id, background_tasks)
        return JSONResponse({"job_id": job.job_id, "status": job.status})

    output_path = await asyncio.to_thread(job_manager.run_job, job.job_id)
    if not output_path:
        raise HTTPException(status_code=500, detail="Job failed to start")
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=output_path.name,
    )


@router.get("/jobs/{job_id}")
async def job_status(job_id: str) -> dict[str, str | None]:
    record = _get_job(job_id)
    response: dict[str, str | None] = {
        "job_id": record.job_id,
        "status": record.status,
        "error": record.error,
        "created_at": record.created_at.isoformat(),
    }
    if record.output_path:
        response["output_path"] = str(record.output_path)
    if record.status == "complete" and record.output_path:
        response["download_url"] = f"/api/jobs/{job_id}/result"
    return response


@router.get("/jobs/{job_id}/result", response_class=FileResponse)
async def job_result(job_id: str) -> FileResponse:
    record = _get_job(job_id)
    if record.status != "complete" or not record.output_path:
        raise HTTPException(status_code=409, detail="Job not finished")
    if not record.output_path.exists():
        raise HTTPException(status_code=404, detail="Rendered file missing")
    return FileResponse(record.output_path, media_type="audio/wav", filename=record.output_path.name)


@router.get("/jobs")
async def list_jobs(limit: int = Query(50, le=200)) -> list[dict[str, str | None]]:
    jobs = job_store.list_jobs(limit=limit)
    return [
        {
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "output_path": str(job.output_path) if job.output_path else None,
        }
        for job in jobs
    ]


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str) -> dict[str, str]:
    job_store.delete_job(job_id)
    return {"job_id": job_id, "status": "deleted"}


def _get_job(job_id: str) -> JobRecord:
    try:
        return job_store.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None


@router.get("/assets", dependencies=[Depends(require_assets_api_key)])
async def get_assets() -> dict[str, list[str]]:
    return list_assets(settings.assets_dir)


@router.post("/assets/upload", dependencies=[Depends(require_assets_api_key)])
async def upload_asset(
    category: Literal["drums", "guitars"],
    overwrite: bool = Query(False),
    file: UploadFile = File(...),
) -> dict[str, str]:
    if file.content_type not in ASSET_ALLOWED_MIME:
        raise HTTPException(status_code=415, detail="Unsupported asset type")
    try:
        path = await save_asset(
            settings.assets_dir,
            category,
            file,
            overwrite=overwrite,
            max_bytes=settings.asset_max_bytes,
        )
    except FileExistsError:
        raise HTTPException(status_code=409, detail="File already exists") from None
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    return {"category": category, "filename": path.name}
