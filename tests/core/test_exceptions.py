"""Test MoFix exception hierarchy — status_code, openai_type, openai_code mappings."""

from __future__ import annotations

import pytest

from app.core.exceptions import (
    CommandNotFoundError,
    InvalidRequestError,
    LLMAuthError,
    LLMNotFoundError,
    LLMQuotaError,
    LLMThinkingError,
    MoFixException,
    PendingConfirmationError,
)


class TestMoFixExceptions:
    """验证每个异常类的 status_code/openai_type/openai_code 属性。"""

    @pytest.mark.parametrize("exc_cls,expected", [
        (MoFixException, {"status_code": 500, "openai_type": "api_error", "openai_code": "internal_error"}),
        (LLMAuthError, {"status_code": 401, "openai_type": "invalid_request_error", "openai_code": "invalid_api_key"}),
        (LLMNotFoundError, {"status_code": 404, "openai_type": "invalid_request_error", "openai_code": "not_found"}),
        (LLMQuotaError, {"status_code": 429, "openai_type": "rate_limit_error", "openai_code": "rate_limit_exceeded"}),
        (LLMThinkingError, {"status_code": 400, "openai_type": "invalid_request_error", "openai_code": "bad_request"}),
        (InvalidRequestError, {"status_code": 400, "openai_type": "invalid_request_error", "openai_code": "invalid_parameter"}),
        (CommandNotFoundError, {"status_code": 400, "openai_type": "invalid_request_error", "openai_code": "invalid_command"}),
        (PendingConfirmationError, {"status_code": 400, "openai_type": "invalid_request_error", "openai_code": "pending_confirmation"}),
    ])
    def test_exception_properties(self, exc_cls, expected):
        exc = exc_cls("test message")
        assert exc.status_code == expected["status_code"]
        assert exc.openai_type == expected["openai_type"]
        assert exc.openai_code == expected["openai_code"]

    def test_exception_can_contain_message(self):
        exc = MoFixException("custom message")
        assert str(exc) == "custom message"

    def test_exception_inheritance(self):
        assert issubclass(LLMAuthError, MoFixException)
        assert issubclass(LLMNotFoundError, MoFixException)
        assert issubclass(LLMQuotaError, MoFixException)
        assert issubclass(LLMThinkingError, MoFixException)
