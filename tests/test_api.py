from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.api import routes
from app.core.config import get_settings
from app.main import app
from app.services.job_store import JobManager, JobStore


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def upload_bytes(tmp_path: Path) -> bytes:
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture(autouse=True)
def isolate_job_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "jobs.sqlite"
    db_url = f"sqlite:///{db_path}"
    store = JobStore(db_url, retention_hours=1)
    manager = JobManager(routes.pipeline, store)
    monkeypatch.setattr(routes, "job_store", store)
    monkeypatch.setattr(routes, "job_manager", manager)
    monkeypatch.setattr(routes.settings, "assets_dir", tmp_path / "assets")
    monkeypatch.setattr(routes.settings, "assets_api_key", "test-key")
    monkeypatch.setattr(routes.settings, "queue_backend", "local")


@pytest.fixture()
def asset_headers() -> dict[str, str]:
    return {"X-Metalizer-Key": "test-key"}


def test_remix_endpoint_returns_audio(client: TestClient, upload_bytes: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "result.wav"
    sf.write(output_path, np.zeros(44100), 44100)

    class DummyPipeline:
        def process(self, _: Path) -> Path:
            return output_path

    monkeypatch.setattr(routes, "pipeline", DummyPipeline())
    routes.job_manager.pipeline = routes.pipeline

    response = client.post(
        "/api/remix",
        files={"file": ("demo.wav", upload_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content[:4] == b"RIFF"


def test_remix_background_job_flow(client: TestClient, upload_bytes: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "result.wav"
    sf.write(output_path, np.zeros(44100), 44100)

    class DummyPipeline:
        def process(self, _: Path) -> Path:
            return output_path

    monkeypatch.setattr(routes, "pipeline", DummyPipeline())
    routes.job_manager.pipeline = routes.pipeline

    response = client.post(
        "/api/remix?background=true",
        files={"file": ("demo.wav", upload_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    job_id = payload["job_id"]

    status = client.get(f"/api/jobs/{job_id}")
    assert status.status_code == 200
    current_status = status.json()["status"]
    assert current_status in {"queued", "processing", "complete"}

    if current_status != "complete":
        routes.job_manager.run_job(job_id)

    completed = client.get(f"/api/jobs/{job_id}")
    assert completed.json()["status"] == "complete"

    download = client.get(f"/api/jobs/{job_id}/result")
    assert download.status_code == 200
    assert download.content[:4] == b"RIFF"


def test_assets_endpoints(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upload_bytes: bytes, asset_headers: dict[str, str]) -> None:
    assets_root = tmp_path / "assets"
    assets_root.mkdir(exist_ok=True)
    monkeypatch.setattr(routes.settings, "assets_dir", assets_root)

    listing = client.get("/api/assets", headers=asset_headers)
    assert listing.status_code == 200
    assert listing.json()["drums"] == []

    response = client.post(
        "/api/assets/upload?category=drums",
        files={"file": ("kick.wav", upload_bytes, "audio/wav")},
        headers=asset_headers,
    )
    assert response.status_code == 200
    assert response.json()["filename"].endswith(".wav")

    listing_after = client.get("/api/assets", headers=asset_headers)
    assert "kick.wav" in listing_after.json()["drums"]


def test_job_list_endpoint(client: TestClient, upload_bytes: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "result.wav"
    sf.write(output_path, np.zeros(44100), 44100)

    class DummyPipeline:
        def process(self, _: Path) -> Path:
            return output_path

    monkeypatch.setattr(routes, "pipeline", DummyPipeline())
    routes.job_manager.pipeline = routes.pipeline

    response = client.post(
        "/api/remix?background=true",
        files={"file": ("demo.wav", upload_bytes, "audio/wav")},
    )
    job_id = response.json()["job_id"]
    routes.job_manager.run_job(job_id)

    jobs_response = client.get("/api/jobs")
    assert jobs_response.status_code == 200
    jobs = jobs_response.json()
    assert any(job["job_id"] == job_id for job in jobs)