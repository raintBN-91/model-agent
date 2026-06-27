"""Test Claude tools — skill name resolution, chunking, warning filter."""

from __future__ import annotations

from unittest.mock import patch

from app.tools.claude_tools import _chunk_text, _filter_warnings, _resolve_skill_name


class TestFilterWarnings:
    def test_removes_error_lines(self):
        text = "INFO: line1\nerror: something bad\nINFO: line2"
        result = _filter_warnings(text)
        assert "error:" not in result

    def test_compresses_multiple_newlines(self):
        text = "a\n\n\n\n\nb"
        result = _filter_warnings(text)
        assert "\n\n\n" not in result

    def test_handles_none_or_non_string(self):
        assert _filter_warnings(None) is None
        assert _filter_warnings(123) == 123

    def test_strips_trailing_newline(self):
        result = _filter_warnings("hello\n\n")
        assert not result.endswith("\n")

    def test_preserves_normal_text(self):
        text = "line1\nline2\nline3"
        result = _filter_warnings(text)
        assert result == text


class TestChunkText:
    def test_short_text_no_chunking(self):
        result = _chunk_text("hello", 10)
        assert result == ["hello"]

    def test_long_line_chunked(self):
        result = _chunk_text("a" * 100, 30)
        assert len(result) > 1
        assert all(len(c) <= 30 for c in result)

    def test_empty_text(self):
        assert _chunk_text("") == []


class TestResolveSkillName:
    def test_direct_match(self):
        assert _resolve_skill_name("verify-agent") == "verify-agent"

    def test_alias_match(self):
        assert _resolve_skill_name("Ascend_Model_Verifier") == "verify-agent"

    def test_unknown_returns_none(self):
        assert _resolve_skill_name("nonexistent-skill") is None
