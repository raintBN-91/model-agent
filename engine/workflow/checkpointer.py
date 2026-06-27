"""Workflow Checkpointer — 工作流状态持久化与恢复。

将 WorkflowPlan + ExecutionContext 序列化为 JSON 文件，
支持中断后恢复执行。
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from server.config import settings


def _get_checkpoint_dir() -> Path:
    """获取 checkpoint 存储目录。"""
    custom = getattr(settings, "workflow_checkpoint_dir", None)
    if custom:
        path = Path(custom)
    else:
        path = Path.home() / ".mofix" / "workflows"
    path.mkdir(parents=True, exist_ok=True)
    return path


class Checkpointer:
    """工作流状态持久化。"""

    def __init__(self, checkpoint_dir: str | Path | None = None):
        self._dir = Path(checkpoint_dir) if checkpoint_dir else _get_checkpoint_dir()

    async def save(self, plan: "WorkflowPlan", ctx: "ExecutionContext") -> None:
        """保存工作流状态到 JSON 文件。"""
        from engine.workflow.types import WorkflowPlan, ExecutionContext

        data = {
            "workflow_id": plan.workflow_id,
            "saved_at": time.time(),
            "saved_at_iso": datetime.now().isoformat(),
            "plan": self._plan_to_dict(plan),
            "context": self._context_to_dict(ctx),
        }

        filepath = self._dir / f"{plan.workflow_id}.json"
        # 原子写：先写临时文件再 rename
        tmp_path = filepath.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        tmp_path.rename(filepath)

    async def load(
        self, workflow_id: str
    ) -> tuple["WorkflowPlan", "ExecutionContext"] | None:
        """从 JSON 文件恢复工作流状态。"""
        from engine.workflow.types import (
            ExecutionContext,
            StepStatus,
            StepType,
            WorkflowPlan,
            WorkflowStep,
        )

        filepath = self._dir / f"{workflow_id}.json"
        if not filepath.is_file():
            return None

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        plan_dict = data.get("plan", {})
        ctx_dict = data.get("context", {})

        # 重建 WorkflowPlan
        steps = []
        for s in plan_dict.get("steps", []):
            parallel_steps = None
            if s.get("parallel_steps"):
                parallel_steps = [
                    WorkflowStep(
                        id=ps["id"],
                        type=StepType(ps.get("type", "claude_skill")),
                        name=ps.get("name", ""),
                        description=ps.get("description", ""),
                        skill_name=ps.get("skill_name"),
                        prompt_template=ps.get("prompt_template", ""),
                        params=ps.get("params", {}),
                        depends_on=ps.get("depends_on", []),
                    )
                    for ps in s["parallel_steps"]
                ]

            step = WorkflowStep(
                id=s["id"],
                type=StepType(s.get("type", "claude_skill")),
                name=s.get("name", ""),
                description=s.get("description", ""),
                skill_name=s.get("skill_name"),
                prompt_template=s.get("prompt_template", ""),
                depends_on=s.get("depends_on", []),
                params=s.get("params", {}),
                max_retries=s.get("max_retries", 2),
                timeout=s.get("timeout", 600),
                condition_expression=s.get("condition_expression"),
                parallel_steps=parallel_steps,
                loop_max_iterations=s.get("loop_max_iterations", 5),
                loop_condition=s.get("loop_condition"),
                status=StepStatus(s.get("status", "pending")),
                result=s.get("result"),
                error=s.get("error"),
                attempts=s.get("attempts", 0),
            )
            steps.append(step)

        plan = WorkflowPlan(
            workflow_id=data["workflow_id"],
            intent=plan_dict.get("intent", ""),
            goal=plan_dict.get("goal", ""),
            steps=steps,
            max_concurrency=plan_dict.get("max_concurrency", 3),
            verify_enabled=plan_dict.get("verify_enabled", True),
        )

        # 重建 ExecutionContext
        ctx = ExecutionContext(
            workflow_id=data["workflow_id"],
            variables=ctx_dict.get("variables", {}),
            step_results=ctx_dict.get("step_results", {}),
            errors=ctx_dict.get("errors", {}),
            metadata=ctx_dict.get("metadata", {}),
        )

        return plan, ctx

    async def list_checkpoints(self) -> list[dict[str, Any]]:
        """列出所有可恢复的工作流。"""
        checkpoints = []
        for f in sorted(self._dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                plan = data.get("plan", {})
                steps = plan.get("steps", [])
                completed = sum(1 for s in steps if s.get("status") == "completed")
                total = len(steps)
                checkpoints.append({
                    "workflow_id": data["workflow_id"],
                    "saved_at": data.get("saved_at_iso", ""),
                    "intent": plan.get("intent", "")[:100],
                    "goal": plan.get("goal", "")[:100],
                    "progress": f"{completed}/{total}",
                    "file": f.name,
                })
            except Exception:
                continue
        return checkpoints

    async def delete(self, workflow_id: str) -> bool:
        """删除指定工作流的 checkpoint。"""
        filepath = self._dir / f"{workflow_id}.json"
        if filepath.is_file():
            filepath.unlink()
            return True
        return False

    # ── 序列化辅助 ──

    def _plan_to_dict(self, plan: "WorkflowPlan") -> dict:
        return {
            "workflow_id": plan.workflow_id,
            "intent": plan.intent,
            "goal": plan.goal,
            "max_concurrency": plan.max_concurrency,
            "verify_enabled": plan.verify_enabled,
            "steps": [
                {
                    "id": s.id,
                    "type": s.type.value,
                    "name": s.name,
                    "description": s.description,
                    "skill_name": s.skill_name,
                    "prompt_template": s.prompt_template,
                    "depends_on": s.depends_on,
                    "params": s.params,
                    "max_retries": s.max_retries,
                    "timeout": s.timeout,
                    "condition_expression": s.condition_expression,
                    "loop_max_iterations": s.loop_max_iterations,
                    "loop_condition": s.loop_condition,
                    "status": s.status.value,
                    "result": s.result,
                    "error": s.error,
                    "attempts": s.attempts,
                    "parallel_steps": [
                        {
                            "id": ps.id,
                            "type": ps.type.value,
                            "name": ps.name,
                            "skill_name": ps.skill_name,
                            "prompt_template": ps.prompt_template,
                            "params": ps.params,
                        }
                        for ps in (s.parallel_steps or [])
                    ] if s.parallel_steps else None,
                }
                for s in plan.steps
            ],
        }

    def _context_to_dict(self, ctx: "ExecutionContext") -> dict:
        return {
            "workflow_id": ctx.workflow_id,
            "variables": ctx.variables,
            "step_results": ctx.step_results,
            "errors": ctx.errors,
            "metadata": ctx.metadata,
        }
