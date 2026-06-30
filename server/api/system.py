"""系统接口 — GET /health。"""

from __future__ import annotations

from fastapi import APIRouter

from server.config import settings
from engine.llm_factory import get_llm_model_name
from engine.registry import get_all_tools
from lts_logger import lts_logger

router = APIRouter()


@router.api_route("/_stcore/health", methods=["GET", "HEAD"])
async def health_check() -> dict:
    lts_logger.log("health_check", status="ok", tools_loaded=len(get_all_tools()))
    return {
        "status": "ok",
        "version": settings.app_version,
        "model": get_llm_model_name(),
        "tools_loaded": len(get_all_tools()),
    }
