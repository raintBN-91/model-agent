"""模型配置修改接口 — POST /v1/chat/config。"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from server.models.config import UpdateModelConfigRequest
from server.services.config_service import update_model_config

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/chat/config")
async def update_model_config_endpoint(
    request: UpdateModelConfigRequest,
) -> dict[str, bool]:
    """修改 Anthropic 模型配置并同步到运行时。"""
    success = update_model_config(
        access_token=request.access_token,
        base_url=request.base_url,
        model=request.model,
    )
    return {"success": success}
