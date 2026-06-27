"""Test chat service — message conversion, SSE format, command execution.

Uses lazy imports because app.services.chat_service imports lts_logger
at module level, which triggers a missing SDK dependency at import time.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _patch_sys_modules():
    """Ensure lts_logger is mocked in sys.modules before any chat_service import."""
    import sys
    from unittest.mock import MagicMock
    mock_mod = MagicMock()
    mock_mod.lts_logger = MagicMock()
    sys.modules.setdefault("lts_logger", mock_mod)


class TestConvertMessages:
    def test_system_message(self):
        from app.services.chat_service import _convert_messages
        from app.models.chat import ChatMessage

        msgs = [ChatMessage(role="system", content="be helpful")]
        result = _convert_messages(msgs)
        assert len(result) == 1
        assert result[0].type == "system"
        assert result[0].content == "be helpful"

    def test_user_message(self):
        from app.services.chat_service import _convert_messages
        from app.models.chat import ChatMessage

        msgs = [ChatMessage(role="user", content="hello")]
        result = _convert_messages(msgs)
        assert len(result) == 1
        assert result[0].type == "human"

    def test_assistant_message(self):
        from app.services.chat_service import _convert_messages
        from app.models.chat import ChatMessage

        msgs = [ChatMessage(role="assistant", content="hi")]
        result = _convert_messages(msgs)
        assert len(result) == 1
        assert result[0].type == "ai"

    def test_tool_message(self):
        from app.services.chat_service import _convert_messages
        from app.models.chat import ChatMessage

        msgs = [ChatMessage(role="tool", content="result", name="search", tool_call_id="call_1")]
        result = _convert_messages(msgs)
        assert len(result) == 1
        assert result[0].type == "tool"

    def test_multiple_messages(self):
        from app.services.chat_service import _convert_messages
        from app.models.chat import ChatMessage

        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="usr"),
            ChatMessage(role="assistant", content="asst"),
        ]
        result = _convert_messages(msgs)
        assert len(result) == 3


class TestBuildSseChunk:
    def test_basic_chunk(self):
        from app.services.chat_service import _build_sse_chunk
        from app.models.chat import DeltaContent

        result = _build_sse_chunk("chunk_1", "model-x", DeltaContent(role="assistant", content="hi"))
        assert result.startswith("data: ")
        assert "chunk_1" in result
        assert "model-x" in result

    def test_with_finish_reason(self):
        from app.services.chat_service import _build_sse_chunk
        from app.models.chat import DeltaContent

        result = _build_sse_chunk("chunk_1", "model-x", DeltaContent(role="assistant", content=""), finish_reason="stop")
        assert '"finish_reason":"stop"' in result

    def test_with_requires_confirmation(self):
        from app.services.chat_service import _build_sse_chunk
        from app.models.chat import DeltaContent

        result = _build_sse_chunk("chunk_1", "m", DeltaContent(role="assistant", content=""), requires_confirmation=True)
        assert '"requires_confirmation":true' in result

    def test_with_brainstorm_state(self):
        from app.services.chat_service import _build_sse_chunk
        from app.models.chat import DeltaContent

        result = _build_sse_chunk("chunk_1", "m", DeltaContent(role="assistant", content=""), brainstorm_state="clarifying")
        assert '"brainstorm_state":"clarifying"' in result


class TestExtractStreamConfig:
    def test_no_c_id(self):
        from app.services.chat_service import _extract_stream_config

        request = MagicMock()
        request.c_id = None
        config, kwargs = _extract_stream_config(request)
        assert config is None

    def test_with_c_id(self):
        from app.services.chat_service import _extract_stream_config

        request = MagicMock()
        request.c_id = "session-1"
        config, kwargs = _extract_stream_config(request)
        assert config == {"configurable": {"thread_id": "session-1"}}


class TestExecuteCommand:
    def _run_async(self, gen):
        """Run async generator to completion, return list of results."""
        loop = asyncio.new_event_loop()
        try:
            results = []
            while True:
                try:
                    results.append(loop.run_until_complete(gen.__anext__()))
                except StopAsyncIteration:
                    break
            return results
        finally:
            loop.close()

    def test_unknown_command(self):
        from app.services.chat_service import _execute_command

        gen = _execute_command("nonexistent", "")
        results = self._run_async(gen)
        assert len(results) >= 2
        assert "[DONE]" in results[-1]

    def test_unknown_command_yields_error(self):
        from app.services.chat_service import _execute_command

        gen = _execute_command("nonexistent", "")
        results = self._run_async(gen)
        assert any("未知命令" in r for r in results)
