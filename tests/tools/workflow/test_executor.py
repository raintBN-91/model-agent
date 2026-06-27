"""Test WorkflowExecutor — ExecutionContext, step status, plan operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.tools.workflow.executor import WorkflowExecutor
from app.tools.workflow.types import (
    ExecutionContext,
    StepStatus,
    StepType,
    WorkflowPlan,
    WorkflowStep,
)


class TestExecutionContext:
    def test_set_step_result(self):
        ctx = ExecutionContext(workflow_id="wf-1")
        ctx.set_step_result("step_1", {"key": "value"})
        assert ctx.step_results["step_1"] == {"key": "value"}
        assert ctx.variables["step_1"] == {"key": "value"}

    def test_get_variable(self):
        ctx = ExecutionContext(workflow_id="wf-1")
        ctx.variables["my_var"] = "hello"
        assert ctx.get("my_var") == "hello"
        assert ctx.get("ctx.my_var") == "hello"

    def test_get_returns_default(self):
        ctx = ExecutionContext(workflow_id="wf-1")
        assert ctx.get("nonexistent") is None
        assert ctx.get("nonexistent", "fallback") == "fallback"

    def test_render_prompt(self):
        ctx = ExecutionContext(workflow_id="wf-1")
        ctx.variables["model_name"] = "Qwen3-8B"
        result = ctx.render_prompt("验证 {ctx.model_name} 的适配")
        assert result == "验证 Qwen3-8B 的适配"

    def test_render_prompt_unknown_key(self):
        ctx = ExecutionContext(workflow_id="wf-1")
        result = ctx.render_prompt("未知 {ctx.unknown_key}")
        assert result == "未知 "  # unknown key replaced with empty string


class TestWorkflowPlan:
    def test_get_ready_steps(self):
        plan = WorkflowPlan(
            intent="test",
            steps=[
                WorkflowStep(id="s1", type="claude_skill", name="S1", depends_on=[]),
                WorkflowStep(id="s2", type="claude_skill", name="S2", depends_on=["s1"]),
            ],
        )
        ready = plan.get_ready_steps(set())
        assert len(ready) == 1
        assert ready[0].id == "s1"

    def test_get_ready_steps_after_completion(self):
        plan = WorkflowPlan(
            intent="test",
            steps=[
                WorkflowStep(id="s1", type="claude_skill", name="S1", depends_on=[]),
                WorkflowStep(id="s2", type="claude_skill", name="S2", depends_on=["s1"]),
            ],
        )
        plan.steps[0].status = StepStatus.COMPLETED
        plan.steps[1].status = StepStatus.PENDING
        ready = plan.get_ready_steps({"s1"})
        assert len(ready) == 1
        assert ready[0].id == "s2"

    def test_is_complete(self):
        plan = WorkflowPlan(intent="test", steps=[
            WorkflowStep(id="s1", type="claude_skill", name="S1"),
        ])
        assert not plan.is_complete()
        plan.steps[0].status = StepStatus.COMPLETED
        assert plan.is_complete()

    def test_has_failed(self):
        plan = WorkflowPlan(intent="test", steps=[
            WorkflowStep(id="s1", type="claude_skill", name="S1"),
        ])
        assert not plan.has_failed()
        plan.steps[0].status = StepStatus.FAILED
        assert plan.has_failed()

    def test_get_step(self):
        plan = WorkflowPlan(intent="test", steps=[
            WorkflowStep(id="s1", type="claude_skill", name="S1"),
        ])
        assert plan.get_step("s1") is plan.steps[0]
        assert plan.get_step("nonexistent") is None


class TestWorkflowExecutor:
    def test_get_step_status(self):
        executor = WorkflowExecutor(checkpointer=MagicMock(), planner=MagicMock())
        plan = WorkflowPlan(
            intent="test",
            steps=[
                WorkflowStep(id="s1", type=StepType.CLAUDE_SKILL, name="Step 1"),
                WorkflowStep(id="s2", type=StepType.LLM_TOOL, name="Step 2", status=StepStatus.RUNNING),
            ],
        )
        statuses = executor.get_step_status(plan)
        assert len(statuses) == 2
        assert statuses[0]["id"] == "s1"
        assert statuses[0]["status"] == "pending"
        assert statuses[1]["id"] == "s2"
        assert statuses[1]["status"] == "running"
        assert statuses[0]["type"] == "claude_skill"
