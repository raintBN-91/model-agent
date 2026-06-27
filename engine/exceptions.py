"""自定义异常 — 自动映射到 OpenAI error 格式。"""

from __future__ import annotations


class MoFixException(Exception):
    """基类，自动映射到 OpenAI error 格式。"""

    openai_type: str = "api_error"
    openai_code: str = "internal_error"
    status_code: int = 500


class LLMAuthError(MoFixException):
    openai_type = "invalid_request_error"
    openai_code = "invalid_api_key"
    status_code = 401


class LLMNotFoundError(MoFixException):
    openai_type = "invalid_request_error"
    openai_code = "not_found"
    status_code = 404


class LLMQuotaError(MoFixException):
    openai_type = "rate_limit_error"
    openai_code = "rate_limit_exceeded"
    status_code = 429


class LLMThinkingError(MoFixException):
    openai_type = "invalid_request_error"
    openai_code = "bad_request"
    status_code = 400


class InvalidRequestError(MoFixException):
    openai_type = "invalid_request_error"
    openai_code = "invalid_parameter"
    status_code = 400


class CommandNotFoundError(MoFixException):
    openai_type = "invalid_request_error"
    openai_code = "invalid_command"
    status_code = 400


class PendingConfirmationError(MoFixException):
    openai_type = "invalid_request_error"
    openai_code = "pending_confirmation"
    status_code = 400
