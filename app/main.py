from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.project_name, version="0.1.0")
app.include_router(api_router, prefix="/api")
