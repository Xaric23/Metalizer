from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile

from app.utils.audio_io import async_persist_upload

SUPPORTED_CATEGORIES = {"drums", "guitars"}


def list_assets(assets_root: Path) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}
    for category in SUPPORTED_CATEGORIES:
        folder = assets_root / category
        if folder.exists():
            payload[category] = sorted(str(path.name) for path in folder.iterdir() if path.is_file())
        else:
            payload[category] = []
    return payload


def get_asset_path(assets_root: Path, category: str, filename: str) -> Path:
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Unsupported category {category}")
    safe_name = Path(filename).name
    folder = assets_root / category
    folder.mkdir(parents=True, exist_ok=True)
    return folder / safe_name


async def save_asset(
    assets_root: Path,
    category: str,
    file: UploadFile,
    *,
    overwrite: bool = False,
    max_bytes: int | None = None,
) -> Path:
    destination = get_asset_path(assets_root, category, file.filename or "sample.wav")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    await async_persist_upload(file, destination, max_bytes=max_bytes)
    return destination
