"""Test Hermes MemoryEngine — extraction, recording, prompt formatting."""

from __future__ import annotations

import pytest


class TestMemoryEngine:
    def test_init_with_store(self, tmp_path):
        from app.tools.experience.store import ExperienceStore
        from app.tools.experience.memory import MemoryEngine
        store = ExperienceStore(data_dir=tmp_path / "experience")
        engine = MemoryEngine(store)
        assert engine is not None

    def test_format_prompt_empty(self, tmp_path):
        from app.tools.experience.store import ExperienceStore
        from app.tools.experience.memory import MemoryEngine
        store = ExperienceStore(data_dir=tmp_path / "experience")
        engine = MemoryEngine(store)
        result = engine.format_prompt("test query")
        assert result is not None
        assert isinstance(result, str)


class TestExtractionHelpers:
    def test_extract_model_from_intent(self):
        from app.tools.experience.memory import _extract_model
        result = _extract_model("verify Qwen3-8B on A2")
        assert "Qwen3" in result

    def test_extract_platform_from_intent(self):
        from app.tools.experience.memory import _extract_platform
        result = _extract_platform("test on A2 platform")
        assert result == "A2"
