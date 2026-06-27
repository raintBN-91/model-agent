"""Test LLM factory — build_llm, build_workflow_llm, handle_api_error."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core.exceptions import (
    LLMAuthError,
    LLMNotFoundError,
    LLMQuotaError,
    LLMThinkingError,
)
from app.core.llm_factory import (
    build_llm,
    build_workflow_llm,
    get_llm_model_name,
    handle_api_error,
)


class TestGetLlmModelName:
    def test_returns_anthropic_model_from_settings(self):
        with patch("app.core.llm_factory.settings") as mock_settings:
            mock_settings.anthropic_model = "claude-3-opus-20240229"
            mock_settings.anthropic_auth_token = "test-token"
            mock_settings.anthropic_base_url = ""
            assert get_llm_model_name() == "claude-3-opus-20240229"

    def test_falls_back_to_default_model(self):
        with patch("app.core.llm_factory.settings") as mock_settings:
            mock_settings.anthropic_model = ""
            mock_settings.anthropic_auth_token = "test-token"
            mock_settings.anthropic_base_url = ""
            assert get_llm_model_name() == "claude-3-5-sonnet-20241022"


class TestBuildLlm:
    def test_build_llm_success(self):
        with patch("app.core.llm_factory.settings") as mock_settings:
            mock_settings.anthropic_auth_token = "test-token"
            mock_settings.anthropic_base_url = "https://api.example.com"
            mock_settings.anthropic_model = "claude-3-5-sonnet-20241022"
            mock_settings.workflow_llm_model = ""
            llm = build_llm(streaming=False)
            assert llm.model == "claude-3-5-sonnet-20241022"

    def test_build_llm_missing_auth_token(self):
        with patch("app.core.llm_factory.settings") as mock_settings:
            mock_settings.anthropic_auth_token = ""
            mock_settings.anthropic_base_url = ""
            mock_settings.anthropic_model = ""
            mock_settings.workflow_llm_model = ""
            with patch("app.core.llm_factory._load_claude_sdk_env", return_value={}):
                with pytest.raises(LLMAuthError, match="ANTHROPIC_AUTH_TOKEN"):
                    build_llm()

    def test_build_llm_reads_claude_sdk_auth_token(self):
        with patch("app.core.llm_factory.settings") as mock_settings:
            mock_settings.anthropic_auth_token = ""
            mock_settings.anthropic_base_url = ""
            mock_settings.anthropic_model = ""
            mock_settings.workflow_llm_model = ""
            claude_env = {
                "ANTHROPIC_AUTH_TOKEN": "kimi-token",
                "ANTHROPIC_BASE_URL": "https://api.kimi.com/coding/",
                "ANTHROPIC_MODEL": "kimi-k2.6",
            }
            with patch("app.core.llm_factory._load_claude_sdk_env", return_value=claude_env):
                llm = build_llm(streaming=False)
                assert llm.model == "kimi-k2.6"


class TestBuildWorkflowLlm:
    def test_workflow_llm_uses_dedicated_model(self):
        with patch("app.core.llm_factory.settings") as mock_settings:
            mock_settings.anthropic_auth_token = "test-token"
            mock_settings.anthropic_base_url = ""
            mock_settings.anthropic_model = "claude-3-5-sonnet-20241022"
            mock_settings.workflow_llm_model = "claude-3-haiku-20240307"
            llm = build_workflow_llm()
            assert llm.model == "claude-3-haiku-20240307"

    def test_workflow_llm_falls_back_to_main_model(self):
        with patch("app.core.llm_factory.settings") as mock_settings:
            mock_settings.anthropic_auth_token = "test-token"
            mock_settings.anthropic_base_url = ""
            mock_settings.anthropic_model = "claude-3-opus-20240229"
            mock_settings.workflow_llm_model = ""
            llm = build_workflow_llm()
            assert llm.model == "claude-3-opus-20240229"

    def test_workflow_llm_low_temperature(self):
        with patch("app.core.llm_factory.settings") as mock_settings:
            mock_settings.anthropic_auth_token = "test-token"
            mock_settings.anthropic_base_url = ""
            mock_settings.anthropic_model = "claude-3-5-sonnet-20241022"
            mock_settings.workflow_llm_model = "planner"
            llm = build_workflow_llm()
            assert llm.temperature == 0.1


class TestHandleApiError:
    def test_401_raises_auth_error(self):
        with pytest.raises(LLMAuthError):
            handle_api_error(Exception("401 Unauthorized"))

    def test_404_raises_not_found_error(self):
        with pytest.raises(LLMNotFoundError):
            handle_api_error(Exception("404 Not Found"))

    def test_429_raises_quota_error(self):
        with pytest.raises(LLMQuotaError):
            handle_api_error(Exception("429 Too Many Requests"))

    def test_insufficient_balance_raises_quota_error(self):
        with pytest.raises(LLMQuotaError):
            handle_api_error(Exception("insufficient balance"))

    def test_thinking_conflict_raises_thinking_error(self):
        with pytest.raises(LLMThinkingError):
            handle_api_error(Exception("reasoning_content not supported"))

    def test_unknown_error_re_raised(self):
        with pytest.raises(ValueError, match="something else"):
            handle_api_error(ValueError("something else"))
