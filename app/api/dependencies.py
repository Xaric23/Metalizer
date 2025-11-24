from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


def require_assets_api_key(
    x_metalizer_key: str | None = Header(default=None, alias="X-Metalizer-Key"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    settings = get_settings()
    if not settings.assets_api_key:
        return
    provided = x_metalizer_key or x_api_key
    if not provided or provided != settings.assets_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")