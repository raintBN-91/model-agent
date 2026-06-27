"""Test chat completion models — request validation only (no FastAPI TestClient needed)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.chat import ChatCompletionRequest, ChatMessage


class TestChatMessage:
    def test_valid_message(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_message_with_tool_fields(self):
        msg = ChatMessage(role="tool", content="result", name="search", tool_call_id="call_1")
        assert msg.tool_call_id == "call_1"

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid_role", content="text")


class TestChatCompletionRequest:
    def test_valid_request(self):
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        assert req.stream is True
        assert req.messages[0].content == "hi"

    def test_empty_messages_raises(self):
        with pytest.raises(ValidationError, match="不能为空"):
            ChatCompletionRequest(messages=[])

    def test_last_message_must_be_user(self):
        with pytest.raises(ValidationError, match="必须是 user"):
            ChatCompletionRequest(messages=[
                ChatMessage(role="user", content="first"),
                ChatMessage(role="assistant", content="last"),
            ])

    def test_system_message_allowed_in_middle(self):
        req = ChatCompletionRequest(messages=[
            ChatMessage(role="system", content="you are helpful"),
            ChatMessage(role="user", content="hi"),
        ])
        assert len(req.messages) == 2

    def test_optional_fields(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            stream=False,
            temperature=0.5,
            max_tokens=1000,
            c_id="test-session",
        )
        assert req.stream is False
        assert req.temperature == 0.5
        assert req.max_tokens == 1000
        assert req.c_id == "test-session"


class TestC2ValidationStrengthening:
    """C2: Pydantic field validation enhancements."""

    def test_temperature_range_accepted(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            temperature=1.5,
        )
        assert req.temperature == 1.5

    def test_temperature_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")],
                temperature=3.0,
            )

    def test_max_tokens_negative_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")],
                max_tokens=-1,
            )

    def test_c_id_too_long_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")],
                c_id="x" * 200,
            )

    def test_empty_content_becomes_none(self):
        msg = ChatMessage(role="user", content="   ")
        assert msg.content is None

    def test_none_content_preserved(self):
        msg = ChatMessage(role="user", content=None)
        assert msg.content is None
