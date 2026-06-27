"""对话接口 — POST /v1/chat/completions (SSE)。"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from engine.llm_factory import build_llm
from server.models.chat import ChatCompletionRequest
from server.services.chat_service import chat_completions_stream, with_heartbeat

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> StreamingResponse:
    trace_id = request.trace_id or uuid.uuid4().hex[:12]
    logger.info(
        "Chat request: trace_id=%s, messages=%d, stream=%s, model=%s",
        trace_id, len(request.messages), request.stream, request.model,
    )

    # 预验证 LLM 配置，让配置错误在 StreamingResponse 发送前抛出
    # （StreamingResponse 内部异常无法转为 HTTP 错误响应）
    build_llm(streaming=True)

    return StreamingResponse(
        with_heartbeat(chat_completions_stream(request)),
        media_type="text/event-stream",
    )
