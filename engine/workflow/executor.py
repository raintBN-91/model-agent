"""DAG 工作流执行引擎。

支持：
- 拓扑排序 + 并行执行独立步骤
- 状态传递（ExecutionContext）
- 失败重试 + 自动 replan
- Checkpoint 恢复
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Awaitable

from engine.workflow.checkpointer import Checkpointer
from engine.workflow.patterns import execute_step_with_pattern
from engine.workflow.planner import DynamicPlanner
from engine.workflow.types import (
    ExecutionContext,
    StepStatus,
    StepType,
    WorkflowPlan,
    WorkflowStep,
)
from server.config import settings

logger = __import__("logging").getLogger(__name__)


class WorkflowExecutor:
    """DAG 工作流执行器。"""

    def __init__(
        self,
        checkpointer: Checkpointer | None = None,
        planner: DynamicPlanner | None = None,
        experience_enabled: bool = True,
    ):
        self._checkpointer = checkpointer or Checkpointer()
        self._planner = planner or DynamicPlanner()
        self._experience_enabled = experience_enabled and settings.hermes_enabled

    async def execute(self, plan: WorkflowPlan, progress_callback: Callable[[dict], Awaitable[None]] | None = None) -> ExecutionContext:
        """执行完整工作流，返回执行上下文。"""
        ctx = ExecutionContext(workflow_id=plan.workflow_id)
        self._progress_callback = progress_callback
        ctx.metadata["max_concurrency"] = plan.max_concurrency

        await self._checkpointer.save(plan, ctx)

        completed: set[str] = set()
        semaphore = asyncio.Semaphore(plan.max_concurrency)

        while not plan.is_complete() and not plan.has_failed():
            ready = plan.get_ready_steps(completed)
            if not ready:
                if not plan.is_complete():
                    # 有可能死锁，检查是否有未完成但是无 ready 步骤
                    pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
                    if pending and not any(
                        all(dep in completed for dep in s.depends_on)
                        for s in pending
                    ):
                        ctx.errors["_deadlock"] = (
                            f"工作流死锁：以下步骤无法满足依赖："
                            f"{[s.id for s in pending]}"
                        )
                        break
                break

            # 并行执行所有 ready 步骤
            tasks = [
                self._execute_with_semaphore(semaphore, step, plan, ctx, completed)
                for step in ready
            ]
            await asyncio.gather(*tasks)

            # 更新 completed 集合
            for step in ready:
                if step.status == StepStatus.COMPLETED:
                    completed.add(step.id)

            await self._checkpointer.save(plan, ctx)

        # ── Hermes Nudge Hook ─────────────────────────────────
        if self._experience_enabled and plan.steps:
            asyncio.create_task(self._run_hermes_post_exec(plan, ctx))
        # ──────────────────────────────────────────────────────

        return ctx

    async def _execute_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        step: WorkflowStep,
        plan: WorkflowPlan,
        ctx: ExecutionContext,
        completed: set[str],
    ) -> None:
        async with semaphore:
            await self._execute_step_with_retry(step, plan, ctx, completed)

    async def _execute_step_with_retry(
        self,
        step: WorkflowStep,
        plan: WorkflowPlan,
        ctx: ExecutionContext,
        completed: set[str],
    ) -> None:
        """执行一个步骤（带重试和 replan）。"""
        logger.info("Executing step: id=%s, name=%s, type=%s", step.id, step.name, step.type.value)
        for attempt in range(step.max_retries + 1):
            step.attempts = attempt + 1
            step.status = StepStatus.RUNNING
            step.started_at = time.time()

            if self._progress_callback:
                await self._progress_callback({"event": "step_start", "step": step.name, "id": step.id, "type": step.type.value})

            try:
                result = await execute_step_with_pattern(
                    step, ctx, plan,
                    progress_callback=self._progress_callback,
                )
                step.status = StepStatus.COMPLETED
                step.result = result
                step.completed_at = time.time()
                ctx.set_step_result(step.id, result)
                if self._progress_callback:
                    await self._progress_callback({"event": "step_end", "step": step.name, "id": step.id, "status": "completed", "result_len": len(str(result))})
                logger.info("Step completed: id=%s, status=%s", step.id, step.status.value)
                return

            except Exception as e:
                step.error = str(e)
                ctx.errors[f"{step.id}.attempt_{attempt + 1}"] = str(e)

                if attempt < step.max_retries:
                    step.status = StepStatus.RETRYING
                    if self._progress_callback:
                        await self._progress_callback({"event": "step_retry", "step": step.name, "id": step.id, "attempt": attempt + 1, "error": str(e)})
                    # 指数退避
                    await asyncio.sleep(2 ** attempt)
                else:
                    step.status = StepStatus.FAILED
                    step.completed_at = time.time()
                    if self._progress_callback:
                        await self._progress_callback({"event": "step_end", "step": step.name, "id": step.id, "status": "failed", "error": step.error})

        # 所有重试失败，尝试 replan
        if step.status == StepStatus.FAILED:
            try:
                new_plan = self._planner.replan(
                    original_input=plan.intent,
                    failed_step=step,
                    error=step.error,
                    context={"context": ctx},
                )
                # 合并新步骤到现有 plan
                for new_step in new_plan.steps:
                    if new_step.id not in {s.id for s in plan.steps}:
                        # 将新步骤的依赖指向失败的步骤
                        if not new_step.depends_on:
                            new_step.depends_on = [step.id]
                        plan.steps.append(new_step)
                # 标记失败步骤为 skipped
                step.status = StepStatus.SKIPPED
            except Exception as replan_err:
                ctx.errors[f"{step.id}.replan"] = str(replan_err)

    async def _run_hermes_post_exec(
        self, plan: WorkflowPlan, ctx: ExecutionContext
    ) -> None:
        """工作流执行完毕后台运行 Hermes 引擎。"""
        try:
            from engine.experience.memory import MemoryEngine
            from engine.experience.skill import SkillEngine
            from engine.experience.nudge import NudgeEngine
            from engine.experience.store import ExperienceStore
            from engine.experience.layers.model_adapt import ModelAdaptLayer

            store = ExperienceStore()
            memory_engine = MemoryEngine(store)
            skill_engine = SkillEngine(store)

            # 1. 提取记忆
            memory = memory_engine.extract(plan, ctx)
            if memory:
                store.append_memory(memory)
                logger.info(f"[Hermes] 已提取记忆 ({memory.get('model', '')})")

            # 2. 评估是否创建 Skill
            skill = skill_engine.evaluate(plan, ctx)
            if skill:
                store.save_skill(skill)
                logger.info(f"[Hermes] 已创建 Skill: {skill['name']}")

            # 3. 后台审查
            nudge_engine = NudgeEngine(store)
            asyncio.create_task(nudge_engine.start_review(plan, ctx))

            # 4. Layer A 适配参数提取
            adapt_layer = ModelAdaptLayer(store)
            adapt_data = adapt_layer.extract_from_workflow(plan, ctx)
            if adapt_data:
                store.save_layer_data("model_adapt", adapt_data)
                logger.info(f"[Hermes] Layer A 已更新")

        except Exception as e:
            logger.warning(f"[Hermes] 后处理异常: {e}")

    async def resume(self, workflow_id: str) -> ExecutionContext | None:
        """从 checkpoint 恢复工作流。"""
        loaded = await self._checkpointer.load(workflow_id)
        if loaded is None:
            return None
        plan, ctx = loaded
        return await self.execute(plan)

    def get_step_status(self, plan: WorkflowPlan) -> list[dict[str, Any]]:
        """获取所有步骤的状态。"""
        return [
            {
                "id": s.id,
                "name": s.name,
                "type": s.type.value,
                "status": s.status.value,
                "error": s.error,
                "attempts": s.attempts,
                "duration": (
                    (s.completed_at - s.started_at)
                    if s.completed_at and s.started_at
                    else None
                ),
            }
            for s in plan.steps
        ]
