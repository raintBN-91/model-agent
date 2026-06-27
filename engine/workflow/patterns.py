"""工作流执行模式。

实现 Claude Code Dynamic Workflows 风格的模式：
- pipeline: 串行依赖执行
- fan_out: 并行独立执行
- verify: 交叉验证
- loop_until: 循环直到条件满足
- classify_and_act: 分类路由
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from engine.workflow.types import (
    ExecutionContext,
    StepStatus,
    StepType,
    WorkflowPlan,
    WorkflowStep,
)


async def execute_step_with_pattern(
    step: WorkflowStep,
    ctx: ExecutionContext,
    plan: WorkflowPlan,
    progress_callback: Any = None,
) -> Any:
    """根据步骤类型执行对应的模式。"""
    if step.type == StepType.PARALLEL:
        return await _execute_parallel(step, ctx, plan, progress_callback)
    elif step.type == StepType.VERIFY:
        return await _execute_verify(step, ctx, plan, progress_callback)
    elif step.type == StepType.CONDITION:
        return await _execute_condition(step, ctx, plan, progress_callback)
    elif step.type == StepType.LOOP:
        return await _execute_loop(step, ctx, plan, progress_callback)
    elif step.type == StepType.CLAUDE_SKILL:
        return await _execute_claude_skill(step, ctx, plan, progress_callback)
    elif step.type == StepType.LLM_TOOL:
        return await _execute_llm_tool(step, ctx, plan, progress_callback)
    elif step.type == StepType.MCP_TOOL:
        return await _execute_mcp_tool(step, ctx, plan, progress_callback)
    else:
        return await _execute_custom(step, ctx, plan, progress_callback)


async def _execute_parallel(
    step: WorkflowStep,
    ctx: ExecutionContext,
    plan: WorkflowPlan,
    progress_callback: Any = None,
) -> list[Any]:
    """并行执行多个子步骤。"""
    if not step.parallel_steps:
        return []

    # 为每个子步骤渲染 prompt
    for sub_step in step.parallel_steps:
        sub_step.prompt_template = ctx.render_prompt(sub_step.prompt_template)

    results = await asyncio.gather(
        *[
            _call_skill_or_tool(ss, ctx, progress_callback=progress_callback)
            for ss in step.parallel_steps
        ],
        return_exceptions=True,
    )

    # 处理异常
    processed = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            processed.append(f"[错误] step_{i}: {r}")
        else:
            processed.append(r)

    return processed


async def _execute_verify(
    step: WorkflowStep,
    ctx: ExecutionContext,
    plan: WorkflowPlan,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """验证步骤：检查前一步的输出。"""
    # 找到被验证的步骤
    target_id = step.depends_on[0] if step.depends_on else None
    target_result = ctx.get(target_id, "") if target_id else ""

    prompt = ctx.render_prompt(step.prompt_template)
    prompt += f"\n\n## 待验证的输出\n\n{target_result}"

    result = await _call_skill_or_tool(
        WorkflowStep(
            id=f"{step.id}_exec",
            type=StepType.CLAUDE_SKILL,
            name=step.name,
            skill_name=step.skill_name,
            prompt_template=prompt,
            params=step.params,
        ),
        ctx,
        progress_callback=progress_callback,
    )

    # 判断是否通过
    result_lower = (result or "").lower()
    passed = any(kw in result_lower for kw in ["通过", "pass", "正确", "valid", "correct", "yes", "consistent"])

    return {
        "passed": passed,
        "verdict": result,
        "verified_step": target_id,
    }


async def _execute_condition(
    step: WorkflowStep,
    ctx: ExecutionContext,
    plan: WorkflowPlan,
    progress_callback: Any = None,
) -> bool:
    """条件判断步骤。"""
    if not step.condition_expression:
        return True

    expr = ctx.render_prompt(step.condition_expression)
    # 简单条件评估：通过 LLM 判断
    prompt = (
        f"根据以下上下文判断条件是否成立。只回答 true 或 false。\n\n"
        f"上下文：{json.dumps(ctx.variables, ensure_ascii=False)}\n\n"
        f"条件：{expr}"
    )

    result = await _call_skill_or_tool(
        WorkflowStep(
            id=f"{step.id}_eval",
            type=StepType.CLAUDE_SKILL,
            name=f"条件判断：{step.name}",
            prompt_template=prompt,
        ),
        ctx,
        progress_callback=progress_callback,
    )

    is_true = (result or "").lower().strip() == "true"
    return is_true


async def _execute_loop(
    step: WorkflowStep,
    ctx: ExecutionContext,
    plan: WorkflowPlan,
    progress_callback: Any = None,
) -> list[Any]:
    """循环执行直到满足条件。"""
    results = []
    for i in range(step.loop_max_iterations):
        ctx.variables[f"{step.id}.iteration"] = i + 1
        ctx.variables[f"{step.id}.last_result"] = results[-1] if results else ""

        result = await _call_skill_or_tool(step, ctx, progress_callback=progress_callback)
        results.append(result)

        # 检查终止条件
        if step.loop_condition:
            cond_prompt = (
                f"根据以下结果判断循环是否终止。如果满足终止条件，回答 true。\n\n"
                f"结果：{json.dumps(result, ensure_ascii=False)}\n\n"
                f"终止条件：{step.loop_condition}"
            )
            cond_result = await _call_skill_or_tool(
                WorkflowStep(
                    id=f"{step.id}_check_{i}",
                    type=StepType.CLAUDE_SKILL,
                    name="循环条件检查",
                    prompt_template=cond_prompt,
                ),
                ctx,
                progress_callback=progress_callback,
            )
            if (cond_result or "").lower().strip() == "true":
                break

    return results


async def _execute_claude_skill(
    step: WorkflowStep,
    ctx: ExecutionContext,
    plan: WorkflowPlan,
    progress_callback: Any = None,
) -> str:
    """调用 Claude Code Skill。"""
    prompt = ctx.render_prompt(step.prompt_template)

    # 立即发送启动反馈，避免 Claude Code 初始化期间用户看不到任何输出
    if progress_callback:
        try:
            await progress_callback({
                "event": "step_output",
                "step": step.name,
                "chunk": "\n> Claude Code 正在初始化 Skill 环境，请稍候...\n",
            })
        except Exception:
            pass

    return await _call_claude_skill(step.skill_name or "", prompt, step.params,
                                    progress_callback=progress_callback, step_name=step.name)


async def _execute_llm_tool(
    step: WorkflowStep,
    ctx: ExecutionContext,
    plan: WorkflowPlan,
    progress_callback: Any = None,
) -> str:
    """调用 LangChain Tool。"""
    prompt = ctx.render_prompt(step.prompt_template)
    return await _call_llm_tool(step.skill_name or "", prompt, step.params, progress_callback=progress_callback)


async def _execute_custom(
    step: WorkflowStep,
    ctx: ExecutionContext,
    plan: WorkflowPlan,
    progress_callback: Any = None,
) -> Any:
    """自定义步骤 — 默认按 claude_skill 处理。"""
    prompt = ctx.render_prompt(step.prompt_template)
    return await _call_claude_skill(step.skill_name or "", prompt, step.params,
                                    progress_callback=progress_callback, step_name=step.name)


async def _execute_mcp_tool(
    step: WorkflowStep,
    ctx: ExecutionContext,
    plan: WorkflowPlan,
    progress_callback: Any = None,
) -> str:
    """调用 MCP 服务器上的外部 Agent 工具。

    step.skill_name → MCP 工具名称
    step.params["mcp_server"] → MCP 服务器名称
    """
    server = step.params.get("mcp_server", "")
    tool = step.skill_name or ""
    prompt = ctx.render_prompt(step.prompt_template)

    if not server:
        return "[MCP] 未指定 mcp_server (在 step.params 中设置)"
    if not tool:
        return "[MCP] 未指定工具名称 (在 step.skill_name 中设置)"

    try:
        from engine.workflow.mcp_integration import get_mcp_manager
        import asyncio
        manager = get_mcp_manager()
        result = await asyncio.wait_for(
            manager.call_tool(server, tool, {"prompt": prompt}),
            timeout=getattr(step, "timeout", 900),
        )
        return result
    except asyncio.TimeoutError:
        return f"[MCP] 调用超时 ({server}.{tool}): >={step.timeout}s（步骤标记为失败）"
    except ImportError as e:
        return f"[MCP] 导入失败: {e}"
    except Exception as e:
        return f"[MCP] 调用失败 ({server}.{tool}): {type(e).__name__}: {e}"


# ── 工具调用封装 ─────────────────────────────────────────────────────

async def _call_skill_or_tool(
    step: WorkflowStep,
    ctx: ExecutionContext,
    progress_callback: Any = None,
) -> Any:
    """根据步骤类型调用对应的执行器。"""
    if step.type == StepType.CLAUDE_SKILL or step.type == StepType.CUSTOM:
        return await _call_claude_skill(
            step.skill_name or "", step.prompt_template, step.params,
            progress_callback=progress_callback,
            step_name=step.name,
        )
    elif step.type == StepType.LLM_TOOL:
        return await _call_llm_tool(
            step.skill_name or "", step.prompt_template, step.params,
            progress_callback=progress_callback,
        )
    elif step.type == StepType.VERIFY:
        return await _call_claude_skill(
            step.skill_name or "", step.prompt_template, step.params,
            progress_callback=progress_callback,
            step_name=step.name,
        )
    else:
        return await _call_claude_skill(
            step.skill_name or "", step.prompt_template, step.params,
            progress_callback=progress_callback,
            step_name=step.name,
        )


async def _call_claude_skill(
    skill_name: str,
    prompt: str,
    params: dict[str, Any] | None = None,
    progress_callback: Any = None,
    step_name: str = "",
) -> str:
    """调用 Claude Code Skill（流式收集结果，支持实时推送）。

    采用与 /verify 命令相同的简单 async for 迭代模式，
    确保流式输出行为一致。心跳改为独立后台任务发送。
    """
    import logging
    _logger = logging.getLogger(__name__)
    try:
        from engine.claude_tools import stream_claude_skill
    except ImportError:
        return f"[workflow] 无法导入 stream_claude_skill"

    chunks: list[str] = []
    _logger.info(f"[workflow] Calling skill: {skill_name or 'default'}, prompt_len={len(prompt)}")

    # 独立心跳任务：每 8 秒发送一次进度提示
    _heartbeat_active = True

    async def _send_heartbeats():
        count = 0
        while _heartbeat_active:
            await asyncio.sleep(8)
            if not _heartbeat_active:
                break
            count += 1
            if progress_callback:
                try:
                    await progress_callback({
                        "event": "step_output",
                        "step": step_name or skill_name,
                        "chunk": f"\n> Claude Code 正在执行中...（已等待 {count * 8}s）\n",
                    })
                except Exception:
                    pass

    heartbeat_task = asyncio.ensure_future(_send_heartbeats())

    try:
        # 使用与 /verify 相同的简单 async for 迭代模式
        async for chunk in stream_claude_skill(skill_name, prompt):
            if chunk:
                chunks.append(chunk)
                if progress_callback:
                    try:
                        await progress_callback({
                            "event": "step_output",
                            "step": step_name or skill_name,
                            "chunk": chunk,
                        })
                    except Exception:
                        pass
                if len(chunks) % 10 == 0:
                    _logger.info(f"[workflow] Skill progress: {skill_name}, chunks={len(chunks)}, last_len={len(chunk)}")
    except Exception as e:
        _logger.error(f"[workflow] Skill call failed: {type(e).__name__}: {e}")
        return f"[workflow] Skill 调用失败: {type(e).__name__}: {e}"
    finally:
        _heartbeat_active = False
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    result = "".join(chunks) if chunks else "(无输出)"
    _logger.info(f"[workflow] Skill done: {skill_name}, total_chunks={len(chunks)}, result_len={len(result)}")
    return result


async def _call_llm_tool(
    tool_name: str,
    prompt: str,
    params: dict[str, Any] | None = None,
    progress_callback: Any = None,
) -> str:
    """调用 LangChain Tool。未找到 tool 时返回 prompt 本身（用于简单输出任务）。"""
    try:
        from engine.registry import get_all_tools
        tools = get_all_tools()
        for t in tools:
            if t.name == tool_name:
                kwargs = {"input": prompt}
                if params:
                    kwargs.update(params)
                result = await t.ainvoke(kwargs)
                return str(result)
        # Tool 不存在时，将 prompt 作为直接输出返回（适用于简单输出/问候等任务）
        return prompt
    except Exception as e:
        return f"[workflow] Tool 调用失败: {type(e).__name__}: {e}"
