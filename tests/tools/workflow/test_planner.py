"""Test DynamicPlanner — JSON parsing, cycle detection, fallback plan."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.tools.workflow.planner import DynamicPlanner, IntentResolver
from app.tools.workflow.types import StepStatus, WorkflowPlan, WorkflowStep


class TestParseResponse:
    def test_plain_json(self):
        planner = DynamicPlanner(llm=MagicMock())
        result = planner._parse_response('{"goal": "test", "steps": []}')
        assert result == {"goal": "test", "steps": []}

    def test_json_with_markdown_fence(self):
        planner = DynamicPlanner(llm=MagicMock())
        result = planner._parse_response('```json\n{"goal": "test", "steps": []}\n```')
        assert result == {"goal": "test", "steps": []}

    def test_json_with_plain_fence(self):
        planner = DynamicPlanner(llm=MagicMock())
        result = planner._parse_response('```\n{"goal": "test", "steps": []}\n```')
        assert result == {"goal": "test", "steps": []}

    def test_nested_json_extraction(self):
        planner = DynamicPlanner(llm=MagicMock())
        result = planner._parse_response('Some text before\n{"goal": "extracted", "steps": [{"id": "s1"}]}\nSome text after')
        assert result == {"goal": "extracted", "steps": [{"id": "s1"}]}

    def test_invalid_json_returns_none(self):
        planner = DynamicPlanner(llm=MagicMock())
        result = planner._parse_response("not json at all")
        assert result is None


class TestFallbackPlan:
    def test_fallback_plan_structure(self):
        planner = DynamicPlanner(llm=MagicMock())
        plan = planner._fallback_plan("test input")
        assert isinstance(plan, WorkflowPlan)
        assert plan.intent == "test input"
        assert len(plan.steps) == 1
        assert plan.steps[0].type.value == "claude_skill"
        assert plan.max_concurrency == 1
        assert plan.verify_enabled is False


class TestCycleDetection:
    def test_no_cycle(self):
        planner = DynamicPlanner(llm=MagicMock())
        plan = WorkflowPlan(
            intent="test",
            steps=[
                WorkflowStep(id="s1", type="claude_skill", name="Step 1", depends_on=[]),
                WorkflowStep(id="s2", type="claude_skill", name="Step 2", depends_on=["s1"]),
            ],
        )
        assert not planner._has_cycle(plan)

    def test_with_cycle(self):
        planner = DynamicPlanner(llm=MagicMock())
        plan = WorkflowPlan(
            intent="test",
            steps=[
                WorkflowStep(id="s1", type="claude_skill", name="Step 1", depends_on=["s2"]),
                WorkflowStep(id="s2", type="claude_skill", name="Step 2", depends_on=["s1"]),
            ],
        )
        assert planner._has_cycle(plan)

    def test_self_referencing_cycle(self):
        planner = DynamicPlanner(llm=MagicMock())
        plan = WorkflowPlan(
            intent="test",
            steps=[
                WorkflowStep(id="s1", type="claude_skill", name="Step 1", depends_on=["s1"]),
            ],
        )
        assert planner._has_cycle(plan)


class TestValidatePlan:
    def test_validate_breaks_cycle(self):
        planner = DynamicPlanner(llm=MagicMock())
        plan = WorkflowPlan(
            intent="test",
            steps=[
                WorkflowStep(id="s1", type="claude_skill", name="Step 1", depends_on=["s2"]),
                WorkflowStep(id="s2", type="claude_skill", name="Step 2", depends_on=["s1"]),
            ],
        )
        planner._validate_plan(plan)
        assert plan.steps[0].depends_on == []
        assert plan.steps[1].depends_on == ["s1"]

    def test_removes_nonexistent_dependency(self):
        planner = DynamicPlanner(llm=MagicMock())
        plan = WorkflowPlan(
            intent="test",
            steps=[
                WorkflowStep(id="s1", type="claude_skill", name="Step 1", depends_on=["nonexistent"]),
            ],
        )
        planner._validate_plan(plan)
        assert plan.steps[0].depends_on == []


class TestIntentResolver:
    def test_construct_and_resolve(self):
        with patch("app.tools.workflow.planner.DynamicPlanner.plan") as mock_plan:
            mock_plan.return_value = WorkflowPlan(intent="test", steps=[])
            resolver = IntentResolver()
            match = resolver.resolve("test")
            assert match.confidence == 0.3
