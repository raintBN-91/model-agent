"""NudgeEngine — 工作流执行后的后台审查与自动改进。

执行完毕后，在后台 fork 分析执行过程，识别可改进的模式，
生成 insight 并自动或建议用户应用改进。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from engine.experience.store import ExperienceStore
from engine.workflow.types import ExecutionContext, StepStatus, WorkflowPlan

logger = logging.getLogger(__name__)


class NudgeEngine:
    """Nudge 引擎 — 执行后审查、insight 生成、自动改进。"""

    def __init__(self, store: ExperienceStore):
        self._store = store

    async def start_review(self, plan: WorkflowPlan, ctx: ExecutionContext) -> None:
        """后台审查执行过程，生成改进建议。"""
        try:
            insights = self.generate_insights(plan, ctx)
            for insight in insights:
                self._store.append_insight(insight)
                if insight.get("auto_appliable"):
                    self.apply_insight(insight)
            if insights:
                logger.info(f"[Nudge] 生成 {len(insights)} 条改进建议")
        except Exception as e:
            logger.warning(f"[Nudge] 审查异常: {e}")

    def generate_insights(self, plan: WorkflowPlan, ctx: ExecutionContext) -> list[dict]:
        """分析执行过程，生成改进建议。"""
        insights: list[dict] = []

        # 1. 检查是否有步骤失败
        failed = [s for s in plan.steps if s.status == StepStatus.FAILED]
        if failed:
            for s in failed:
                insights.append({
                    "type": "error_pattern",
                    "description": f"步骤 {s.id} ({s.name}) 失败: {s.error[:100] if s.error else '未知错误'}",
                    "auto_appliable": False,
                    "plan_id": plan.workflow_id,
                    "step_id": s.id,
                    "error": s.error,
                })

        # 2. 检查是否有步骤重试成功
        retried = [s for s in plan.steps if s.attempts > 1 and s.status == StepStatus.COMPLETED]
        if retried:
            for s in retried:
                insights.append({
                    "type": "retry_success",
                    "description": f"步骤 {s.id} 重试 {s.attempts - 1} 次后成功，考虑增加资源或超时",
                    "auto_appliable": True,
                    "plan_id": plan.workflow_id,
                    "step_id": s.id,
                })

        # 3. 检查是否有长时间运行的步骤
        for s in plan.steps:
            if s.started_at is not None and s.completed_at is not None:
                duration = s.completed_at - s.started_at
                if duration > 300 and s.status == StepStatus.COMPLETED:
                    insights.append({
                        "type": "long_step",
                        "description": f"步骤 {s.id} ({s.name}) 耗时 {duration:.0f}s，考虑拆分或优化",
                        "auto_appliable": False,
                        "plan_id": plan.workflow_id,
                        "step_id": s.id,
                        "duration": duration,
                    })

        return insights

    def apply_insight(self, insight: dict) -> bool:
        """应用可自动应用的改进建议。"""
        if not insight.get("auto_appliable"):
            return False
        self._store.mark_insight_applied(insight.get("id", ""))
        return True

    def suggest_replan(
        self, failed_step: "WorkflowStep", error: str
    ) -> dict | None:
        """根据历史错误模式建议 replan 策略。"""
        insights = self._store.get_insights(limit=50)
        for ins in insights:
            if ins.get("type") == "error_pattern" and ins.get("error"):
                if error[:50].lower() in (ins["error"] or "").lower():
                    return {
                        "suggested_action": "retry_with_different_tool",
                        "reason": "历史中类似的错误在切换工具后恢复",
                        "source_insight": ins["id"],
                    }
        return None
