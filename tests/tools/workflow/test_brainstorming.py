"""Test Brainstorming — state enum, session lifecycle, JSON parsing, ambiguity."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.tools.workflow.brainstorming import (
    BrainstormProposal,
    BrainstormQuestion,
    BrainstormState,
    BrainstormingSession,
    _keyword_ambiguity_fallback,
)


class TestBrainstormState:
    def test_enum_values(self):
        assert BrainstormState.INITIAL.value == "initial"
        assert BrainstormState.CLARIFYING.value == "clarifying"
        assert BrainstormState.PROPOSING.value == "proposing"
        assert BrainstormState.REVIEWING.value == "reviewing"
        assert BrainstormState.APPROVED.value == "approved"
        assert BrainstormState.COMPLETED.value == "completed"
        assert BrainstormState.REJECTED.value == "rejected"

    def test_state_count(self):
        assert len(BrainstormState) == 7


class TestBrainstormQuestion:
    def test_defaults(self):
        q = BrainstormQuestion(question_id="q1", question="test?")
        assert q.options == []
        assert q.allow_custom is True
        assert q.dimension == ""

    def test_with_options(self):
        q = BrainstormQuestion(question_id="q1", question="which?", options=["A", "B"], dimension="platform")
        assert len(q.options) == 2
        assert q.dimension == "platform"


class TestBrainstormProposal:
    def test_defaults(self):
        p = BrainstormProposal(title="Test")
        assert p.description == ""
        assert p.pros == []
        assert p.cons == []
        assert p.estimated_effort == ""


class TestParseJsonResponse:
    def test_plain_json(self):
        session = BrainstormingSession(llm=MagicMock())
        result = session._parse_json_response('{"type": "question", "assessment": "clarify_needed"}')
        assert result["type"] == "question"
        assert result["assessment"] == "clarify_needed"

    def test_json_with_fence(self):
        session = BrainstormingSession(llm=MagicMock())
        result = session._parse_json_response('```json\n{"type": "proposals", "assessment": "ready_for_proposals"}\n```')
        assert result["type"] == "proposals"

    def test_json_with_plain_fence(self):
        session = BrainstormingSession(llm=MagicMock())
        result = session._parse_json_response('```\n{"type": "design_doc", "assessment": "ready_for_review"}\n```')
        assert result["type"] == "design_doc"

    def test_invalid_raises(self):
        session = BrainstormingSession(llm=MagicMock())
        with pytest.raises(ValueError):
            session._parse_json_response("not json")


class TestBrainstormingSession:
    def test_init_defaults(self):
        session = BrainstormingSession(llm=MagicMock())
        assert session.state == BrainstormState.INITIAL
        assert session.user_input == ""
        assert session.clarified_dimensions == {}
        assert session.proposals == []
        assert session.design_doc is None

    def test_start_returns_dict_with_question(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"type": "question", "question": {"text": "which platform?", "options": ["A2", "A3"]}, "assessment": "clarify_needed"}'
        mock_llm.invoke.return_value = mock_response
        session = BrainstormingSession(llm=mock_llm)
        result = session.start("verify model")
        assert result["type"] == "question"
        assert isinstance(result["question"], BrainstormQuestion)
        assert result["question"].question == "which platform?"
        assert session.state == BrainstormState.CLARIFYING

    def test_answer_moves_to_proposals(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content='{"type": "proposals", "proposals": [{"title": "Plan A", "description": "desc"}], "assessment": "ready_for_proposals"}'),
        ]
        session = BrainstormingSession(llm=mock_llm)
        session.state = BrainstormState.CLARIFYING
        session.history = [
            {"role": "assistant", "type": "question", "dimension": "target_platform", "content": "which?"}
        ]
        result = session.answer("q1", "A2")
        assert result["type"] == "proposals"
        assert len(result["proposals"]) == 1
        assert session.state == BrainstormState.PROPOSING

    def test_approve_completes(self):
        session = BrainstormingSession(llm=MagicMock())
        session.state = BrainstormState.REVIEWING
        session.design_doc = MagicMock()
        session.user_input = "test"
        result = session.approve()
        assert result["type"] == "complete"
        assert "context" in result
        assert session.state == BrainstormState.COMPLETED

    def test_reject_with_reason(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"type": "question", "question": {"text": "more info?"}, "assessment": "clarify_needed"}')
        session = BrainstormingSession(llm=mock_llm)
        session.state = BrainstormState.REVIEWING
        result = session.reject("方案不合适")
        assert result["type"] == "question"

    def test_to_dict_roundtrip(self):
        session = BrainstormingSession(session_id="bs-test", llm=MagicMock())
        session.state = BrainstormState.CLARIFYING
        session.user_input = "test input"
        d = session.to_dict()
        assert d["session_id"] == "bs-test"
        assert d["state"] == "clarifying"
        restored = BrainstormingSession.from_dict(d)
        assert restored.session_id == "bs-test"
        assert restored.state == BrainstormState.CLARIFYING
        assert restored.user_input == "test input"

    def test_fallback_response_initial(self):
        session = BrainstormingSession(llm=MagicMock())
        session.state = BrainstormState.INITIAL
        result = session._fallback_response()
        assert result["type"] == "question"
        assert result["question"]["dimension"] == "target_platform"

    def test_fallback_response_with_platform_no_model(self):
        session = BrainstormingSession(llm=MagicMock())
        session.state = BrainstormState.CLARIFYING
        session.clarified_dimensions = {"target_platform": "A2"}
        result = session._fallback_response()
        assert result["type"] == "question"
        assert result["question"]["dimension"] == "model_name"

    def test_fallback_response_with_model_no_framework(self):
        session = BrainstormingSession(llm=MagicMock())
        session.state = BrainstormState.CLARIFYING
        session.clarified_dimensions = {"target_platform": "A2", "model_name": "Qwen3-8B"}
        result = session._fallback_response()
        assert result["type"] == "question"
        assert result["question"]["dimension"] == "inference_framework"

    def test_fallback_response_all_dimensions(self):
        session = BrainstormingSession(llm=MagicMock())
        session.state = BrainstormState.CLARIFYING
        session.clarified_dimensions = {"target_platform": "A2", "model_name": "Qwen", "inference_framework": "vLLM"}
        result = session._fallback_response()
        assert result["type"] == "proposals"


class TestKeywordAmbiguityFallback:
    def test_platform_detected(self):
        score, missing = _keyword_ambiguity_fallback("test on ascend using vllm")
        assert "target_platform" not in missing

    def test_vague_input(self):
        score, missing = _keyword_ambiguity_fallback("help me optimize the model")
        assert "target_platform" in missing
        assert "model_name" in missing
        assert "inference_framework" in missing
        assert 0 < score <= 1.0

    def test_partial_model_name(self):
        score, missing = _keyword_ambiguity_fallback("test llama performance")
        assert "model_name" not in missing
        assert "target_platform" in missing
