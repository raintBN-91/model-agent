"""全局异常处理器 — OpenAI 格式。"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from engine.exceptions import MoFixException
from lts_logger import lts_logger


async def mofix_exception_handler(request: Request, exc: MoFixException) -> JSONResponse:
    lts_logger.log(
        "api_error",
        error_type=exc.__class__.__name__,
        detail=str(exc),
        status_code=exc.status_code,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc),
                "type": exc.openai_type,
                "param": None,
                "code": exc.openai_code,
            }
        },
    )
