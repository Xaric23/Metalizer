from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from fastapi import UploadFile


class TempAudioFile:
    """Context manager to persist uploaded audio to a temp path."""

    def __init__(self, suffix: str = ".wav") -> None:
        self._suffix = suffix
        self._tmp_dir: TemporaryDirectory[str] | None = None
        self.path: Path | None = None

    def __enter__(self) -> "TempAudioFile":
        self._tmp_dir = TemporaryDirectory(prefix="metalizer_")
        base_dir = Path(self._tmp_dir.name)
        self.path = base_dir / f"input{self._suffix}"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._tmp_dir:
            self._tmp_dir.cleanup()
            self.path = None

    def write(self, data: bytes) -> Path:
        if not self.path:
            raise RuntimeError("TempAudioFile not initialized")
        self.path.write_bytes(data)
        return self.path


def chunk_reader(file: UploadFile, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    """Yield chunks from an UploadFile stream."""

    while True:
        chunk = file.file.read(chunk_size)
        if not chunk:
            break
        yield chunk


def persist_upload(file: UploadFile, destination: Path) -> Path:
    """Save an UploadFile to disk in a streaming-safe way."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in chunk_reader(file):
            handle.write(chunk)
    return destination


async def async_persist_upload(
    file: UploadFile,
    destination: Path,
    *,
    chunk_size: int = 1024 * 1024,
    max_bytes: int | None = None,
) -> Path:
    """Asynchronously stream an UploadFile to disk without buffering it all in memory."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise ValueError("File exceeds allowed size")
                handle.write(chunk)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise
    finally:
        await file.close()
    return destination
