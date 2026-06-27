from __future__ import annotations
"""FastAPI 入口。"""


from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api import chat, config, system, eval as eval_routes
from server.config import settings
from engine.exceptions import MoFixException
from server.middleware.error_handler import mofix_exception_handler
from commands.registry import Command, TriggerMode, cmd_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：注册命令、启动时初始化、关闭时清理。"""
    # 挂载本地日志采集（用于 Eval 系统历史分析）
    try:
        import importlib.util
        _eval_path = Path(__file__).resolve().parent.parent / "eval" / "v0.1.0" / "analyzers" / "enable_local_logging.py"
        if _eval_path.exists():
            _spec = importlib.util.spec_from_file_location("enable_local_logging", str(_eval_path))
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _patch_lts_logger = _mod.patch_lts_logger
            logs_dir = Path(__file__).resolve().parent.parent / "logs" / "events"
            _patch_lts_logger(log_dir=str(logs_dir))
    except Exception:
        pass  # eval 模块未部署时不阻塞服务启动

    from engine.claude_tools import stream_claude_skill

    # 注册 /claude 命令
    async def claude_handler(args: str) -> AsyncIterator[str]:
        parts = args.split(maxsplit=1)
        skill_name = parts[0] if parts else ""
        prompt = parts[1] if len(parts) > 1 else ""
        try:
            async for chunk in stream_claude_skill(skill_name, prompt):
                yield chunk
        except Exception as e:
            yield f"[claude-code] 调用失败：{type(e).__name__}: {e}"

    cmd_registry.register(
        Command(
            name="claude",
            description="调用 Claude Code Skill 执行昇腾模型适配、验证、优化等深度任务",
            trigger_modes=[
                TriggerMode.USER_TRIGGER,
                TriggerMode.LLM_RECOMMEND,
                TriggerMode.LLM_AUTO,
            ],
            handler=claude_handler,
            example="/claude verify-agent 帮我验证 Qwen3-8B 在 A2 上的适配",
        )
    )

    # 注册 /verify 命令：固定调用 verify-agent skill
    async def verify_handler(args: str) -> AsyncIterator[str]:
        try:
            async for chunk in stream_claude_skill("verify-agent", args):
                yield chunk
        except Exception as e:
            yield f"[verify] 调用失败：{type(e).__name__}: {e}"

    cmd_registry.register(
        Command(
            name="verify",
            description="调用 Claude Code verify-agent Skill 执行昇腾模型适配验证",
            trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND],
            handler=verify_handler,
            example="/verify 帮我验证 Qwen3-8B 在 A2 上的适配",
        )
    )

    # 注册 /adapt 命令：固定调用 adapt-agent skill
    async def adapt_handler(args: str) -> AsyncIterator[str]:
        try:
            async for chunk in stream_claude_skill("adapt-agent", args):
                yield chunk
        except Exception as e:
            yield f"[adapt] 调用失败：{type(e).__name__}: {e}"

    cmd_registry.register(
        Command(
            name="adapt",
            description="调用 Claude Code adapt-agent Skill 执行昇腾模型适配",
            trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND],
            handler=adapt_handler,
            example="/adapt 帮我适配 Qwen3-8B 到 A2 并使用 vLLM-Ascend",
        )
    )

    # 注册 /optimize 命令：固定调用 optimizer-agent skill
    async def optimize_handler(args: str) -> AsyncIterator[str]:
        try:
            async for chunk in stream_claude_skill("optimizer-agent", args):
                yield chunk
        except Exception as e:
            yield f"[optimize] 调用失败：{type(e).__name__}: {e}"

    cmd_registry.register(
        Command(
            name="optimize",
            description="调用 Claude Code optimizer-agent Skill 执行 vLLM-Ascend 性能调优",
            trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND],
            handler=optimize_handler,
            example="/optimize 帮我优化 Qwen3-8B 在 A2 上的推理性能",
        )
    )

    # 注册 /quantify 命令：固定调用 quantify-agent skill
    async def quantify_handler(args: str) -> AsyncIterator[str]:
        try:
            async for chunk in stream_claude_skill("quantify-agent", args):
                yield chunk
        except Exception as e:
            yield f"[quantify] 调用失败：{type(e).__name__}: {e}"

    cmd_registry.register(
        Command(
            name="quantify",
            description="调用 Claude Code quantify-agent Skill 执行昇腾推理工具链量化",
            trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND],
            handler=quantify_handler,
            example="/quantify 帮我量化 Qwen3-8B 用于昇腾推理",
        )
    )

    # 注册 /commit 命令：固定调用 gitcode-publish skill
    async def commit_handler(args: str) -> AsyncIterator[str]:
        try:
            async for chunk in stream_claude_skill("gitcode-publish", args):
                yield chunk
        except Exception as e:
            yield f"[commit] 调用失败：{type(e).__name__}: {e}"

    cmd_registry.register(
        Command(
            name="commit",
            description="调用 Claude Code gitcode-publish Skill 执行代码发布与提交",
            trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND],
            handler=commit_handler,
            example="/commit 帮我发布当前代码变更到 GitCode",
        )
    )

    # 注册 /search 命令：调用 search_tools
    from engine.search_tools import search_ascend_models

    async def search_handler(args: str) -> str:
        try:
            return search_ascend_models.invoke({"input": args})
        except Exception as e:
            return f"[search] 调用失败：{type(e).__name__}: {e}"

    cmd_registry.register(
        Command(
            name="search",
            description="调用 search-agent 从 GitCode 昇腾模型库搜索已适配模型",
            trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND],
            handler=search_handler,
            example="/search Qwen3-14B 在 A2 上的适配",
        )
    )

    # 注册 /ai4s 命令：固定调用 ai4s-main skill
    async def ai4s_handler(args: str) -> AsyncIterator[str]:
        try:
            async for chunk in stream_claude_skill("ai4s-main", args):
                yield chunk
        except Exception as e:
            yield f"[ai4s] 调用失败：{type(e).__name__}: {e}"

    cmd_registry.register(
        Command(
            name="ai4s",
            description="调用 Claude Code ai4s-main Skill 处理 AI for Science 昇腾 NPU 任务（精度对齐、Profiling、性能调优、模型迁移等）",
            trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND],
            handler=ai4s_handler,
            example="/ai4s 帮我将 Boltz2 模型迁移到昇腾 NPU 并做精度对齐",
        )
    )

    # 注册 /pta 命令：调用 PTA Agent，支持基于 c_id 的多轮对话记忆
    from engine.pta_tools import stream_pta_query

    async def pta_handler(args: str, c_id: str | None = None) -> AsyncIterator[str]:
        try:
            # 将 c_id（会话 ID）透传给 stream_pta_query，
            # 使 PTA Agent 能读取并保存该会话的历史上下文。
            async for chunk in stream_pta_query(args, thread_id=c_id):
                yield chunk
        except Exception as e:
            yield f"[pta] 调用失败：{type(e).__name__}: {e}"

    cmd_registry.register(
        Command(
            name="pta",
            description="调用 PTA Agent 执行 PyTorch-Ascend (torch_npu) 流式问答，支持同一会话多轮记忆",
            trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND],
            handler=pta_handler,
            example="/pta torchair 编译流程是什么",
        )
    )

    # 注册 /deploy 命令：直接通过 Claude 执行部署相关 prompt
    async def deploy_handler(args: str) -> AsyncIterator[str]:
        if not args.strip():
            yield "请提供部署相关的 prompt，例如：\n"
            yield "- `/deploy 检查当前系统环境`\n"
            yield "- `/deploy 部署 Qwen3.5-0.8B 到 Ascend NPU`\n"
            yield "- `/deploy 初始化 vLLM-Ascend 服务`\n"
            return
        try:
            async for chunk in stream_claude_skill("", args):
                yield chunk
        except Exception as e:
            yield f"[deploy] 调用失败：{type(e).__name__}: {e}"

    cmd_registry.register(
        Command(
            name="deploy",
            description="直接通过 Claude 执行部署相关的 prompt",
            trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND],
            handler=deploy_handler,
            example="/deploy 帮我部署 Boltz2 到昇腾 NPU",
        )
    )

    # 注册 /doc 命令：优先调用 doc-agent 工具，未匹配时回退到 Claude
    from engine.doc_tools import (
        doc_agent_example_qwen3_8b,
        doc_agent_example_qwen35_flow,
        doc_agent_generate_local,
        doc_agent_one_click_generate,
        doc_agent_sglang_adapter_docs,
        doc_agent_verl_adapter_docs,
        doc_agent_vllm_adapter_docs,
    )

    async def doc_handler(args: str) -> AsyncIterator[str]:
        parts = args.strip().split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        result = ""
        try:
            if sub in ("vllm", "vllm-ascend"):
                result = doc_agent_vllm_adapter_docs.invoke({})
            elif sub in ("sglang",):
                result = doc_agent_sglang_adapter_docs.invoke({})
            elif sub in ("verl",):
                result = doc_agent_verl_adapter_docs.invoke({})
            elif sub in ("qwen3-8b", "qwen3_8b", "qwen38b"):
                result = doc_agent_example_qwen3_8b.invoke({})
            elif sub in ("qwen3.5-27b", "qwen35", "qwen3.5"):
                result = doc_agent_example_qwen35_flow.invoke({})
            elif sub in ("generate", "local"):
                manifest = rest.strip() or "manifest.discovered.yaml"
                result = doc_agent_generate_local.invoke({"manifest_filename": manifest})
            elif sub in ("discover", "one-click", "one_click"):
                result = doc_agent_one_click_generate.invoke({"extra_args": rest})
            else:
                # 不认识的子命令已在 chat_service 层透传给 LLM，此处不会走到
                yield f"[doc] 未知子命令 `{sub}`。可用：vllm、sglang、verl、qwen3-8b、qwen3.5-27b、generate、discover"
                return
        except Exception as e:
            result = f"[doc] 调用失败：{type(e).__name__}: {e}"

        # 同步结果需要模拟流式输出（按行分段）
        for line in result.splitlines(keepends=True):
            yield line

    cmd_registry.register(
        Command(
            name="doc",
            description="调用 doc-agent 工具执行文档生成与查询：vllm/sglang/verl/qwen3-8b/qwen3.5-27b/generate/discover",
            trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND, TriggerMode.LLM_AUTO],
            handler=doc_handler,
            example="/doc vllm    | /doc qwen3-8b    | /doc generate    | /doc discover --only Qwen3.5-27B",
        )
    )

    # 注册 /workflow 命令：使用 Dynamic Workflow Engine（LLM 规划 + DAG 执行）
    from engine.workflow import DynamicPlanner, WorkflowExecutor

    async def _format_plan_text(plan) -> str:
        """将 WorkflowPlan 格式化为可读文本。"""
        lines = [
            f"## Dynamic Workflow Plan",
            f"",
            f"**Goal**: {plan.goal or 'N/A'}",
            f"**Steps**: {len(plan.steps)}",
            f"**Max Concurrency**: {plan.max_concurrency}",
            f"**Verify Enabled**: {plan.verify_enabled}",
            f"",
            f"### Execution DAG ({len(plan.steps)} steps)",
            f"",
        ]
        for i, step in enumerate(plan.steps, 1):
            deps = ", ".join(step.depends_on) or "none"
            lines.append(f"  {i}. `{step.id}` **{step.name}** ({step.type.value})")
            lines.append(f"     → skill: `{step.skill_name or 'N/A'}`")
            lines.append(f"     → depends_on: [{deps}]")
            if step.type.value == "mcp_tool":
                mcp_server = step.params.get("mcp_server", "?")
                lines.append(f"     → mcp_server: `{mcp_server}`")
            if step.type.value == "parallel" and step.parallel_steps:
                for ps in step.parallel_steps:
                    lines.append(f"       · {ps.id}: {ps.name} ({ps.skill_name or 'N/A'})")
        return "\n".join(lines)

    async def workflow_handler(args: str) -> AsyncIterator[str]:
        # Step 1: LLM 规划工作流
        yield "**AI 正在分析您的需求并规划工作流...**\n\n"
        planner = DynamicPlanner()
        plan = await planner.aplan(args)

        # Step 2: 展示计划
        plan_text = await _format_plan_text(plan)
        yield plan_text
        yield "\n\n---\n\n"

        if not plan.steps:
            yield "未能生成工作流步骤，将直接处理您的需求。\n\n"
            async for chunk in stream_claude_skill("", args):
                yield chunk
            return

        # Step 3: DAG 执行（带进度回调，实时推送 SSE）
        yield "**开始执行工作流...**\n\n"

        import asyncio as _asyncio
        _progress_queue: _asyncio.Queue = _asyncio.Queue()

        async def _on_progress(event: dict):
            await _progress_queue.put(event)

        executor = WorkflowExecutor()

        # 启动后台执行任务
        _exec_task = _asyncio.ensure_future(
            executor.execute(plan, _on_progress)
        )

        # 在执行期间消费进度事件并实时 yield
        while not _exec_task.done():
            try:
                event = await _asyncio.wait_for(_progress_queue.get(), timeout=0.5)
                evt = event["event"]
                if evt == "step_start":
                    yield f"\n**[RUNNING] {event['step']}** (id: {event['id']}, type: {event['type']})\n"
                elif evt == "step_output":
                    yield event.get("chunk", "")
                elif evt == "step_end":
                    icon = "[PASS]" if event["status"] == "completed" else "[FAIL]"
                    yield f"\n**[{icon}] {event['step']}** (id: {event['id']}) 完成\n"
                    if event.get("error"):
                        yield f"  错误: {event['error']}\n"
                elif evt == "step_retry":
                    yield f"**[RETRY] {event['step']}** (id: {event['id']}) 第{event['attempt']}次重试\n"
            except _asyncio.TimeoutError:
                pass

        ctx = await _exec_task

        # Step 4: 输出结果
        for step in plan.steps:
            if step.status.value == "completed" and step.id in ctx.step_results:
                yield f"\n[PASS] **{step.name}** (id: {step.id}) 完成\n\n"
                result = ctx.step_results[step.id]
                result_str = str(result)
                # 控制输出长度避免 SSE 过大
                if len(result_str) > 3000:
                    result_str = result_str[:3000] + "\n\n...（输出已截断）"
                # MCP 工具步骤特殊展示
                if step.type.value == "mcp_tool":
                    mcp_server = step.params.get("mcp_server", "")
                    yield f"**MCP 调用**: {step.skill_name} ({mcp_server})\n"
                    yield f"{result_str[:2000]}\n"
                # 验证步骤特殊展示
                elif step.type.value == "verify":
                    if isinstance(result, dict):
                        passed = result.get("passed", False)
                        yield f"**验证结果**: {'[PASS] 通过' if passed else '[FAIL] 未通过'}\n"
                        yield f"**详情**: {result.get('verdict', '')[:500]}\n"
                    else:
                        yield f"{result_str[:1000]}\n"
                else:
                    yield f"{result_str[:2000]}\n"
            elif step.status.value == "failed":
                yield f"\n[FAIL] **{step.name}** (id: {step.id}) 失败: {step.error}\n\n"
            elif step.status.value == "skipped":
                yield f"\n[SKIP] **{step.name}** (id: {step.id}) 已跳过\n\n"

        # Step 5: 端到端报告
        total = len(plan.steps)
        completed = sum(1 for s in plan.steps if s.status.value == "completed")
        failed = sum(1 for s in plan.steps if s.status.value == "failed")
        skipped = sum(1 for s in plan.steps if s.status.value == "skipped")

        yield "\n\n---\n\n"
        yield "## 📊 端到端工作流执行报告\n\n"

        # 总体摘要
        yield f"| 指标 | 值 |\n"
        yield f"|------|-----|\n"
        yield f"| **工作流目标** | {plan.goal} |\n"
        yield f"| **总步骤数** | {total} |\n"
        yield f"| **成功完成** | {completed} |\n"
        if failed:
            yield f"| **失败** | {failed} |\n"
        if skipped:
            yield f"| **已跳过** | {skipped} |\n"
        yield f"| **最大并发** | {plan.max_concurrency} |\n"
        yield "\n"

        # 步骤详情表
        yield "### 步骤执行详情\n\n"
        yield f"| 步骤ID | 名称 | 类型 | 状态 | 说明 |\n"
        yield f"|--------|------|------|------|------|\n"
        for step in plan.steps:
            status_icon = {
                "completed": "[PASS]",
                "failed": "[FAIL]",
                "skipped": "[SKIP]",
                "pending": "[PEND]",
                "running": "[RUN]",
            }.get(step.status.value, step.status.value)
            desc = (step.description or step.name)[:60]
            yield f"| {step.id} | {step.name} | {step.type.value} | {status_icon} | {desc} |\n"
        yield "\n"

        # 详细结果
        yield "### 详细结果\n\n"
        for step in plan.steps:
            if step.status.value == "completed" and step.id in ctx.step_results:
                result = ctx.step_results[step.id]
                result_str = str(result)
                if len(result_str) > 2000:
                    result_str = result_str[:2000] + "\n\n...（输出已截断）"
                yield f"#### ✅ {step.name} (id: {step.id})\n\n"
                if step.type.value == "mcp_tool":
                    yield f"**MCP 调用**: {step.skill_name} ({step.params.get('mcp_server', 'N/A')})\n\n"
                yield f"{result_str[:2000]}\n\n"
            elif step.status.value == "failed":
                yield f"#### ❌ {step.name} (id: {step.id})\n\n"
                yield f"**错误**: {step.error}\n\n"
            elif step.status.value == "skipped":
                yield f"#### ⏭️ {step.name} (id: {step.id})\n\n"
                yield f"步骤已跳过。\n\n"

        # 结束标记
        yield "---\n"
        yield f"**工作流执行完毕**: {completed}/{total} 步骤完成"
        if failed:
            yield f"，{failed} 步骤失败"
        yield "\n"

    cmd_registry.register(
        Command(
            name="workflow",
            description="使用 Dynamic Workflow Engine 解析意图并自动执行多步骤工作流（LLM 规划 + DAG 执行）",
            trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND],
            handler=workflow_handler,
            example="/workflow 验证并优化 Qwen3-8B 在 A2 上的适配",
        )
    )

    # ── 注册 /experience 命令 ─────────────────────────────────────
    async def experience_handler(args: str) -> AsyncIterator[str]:
        """查询和管理 Hermes 自演进引擎的经验数据。"""
        parts = args.strip().split()
        sub = parts[0].lower() if parts else "stats"

        from engine.experience.store import ExperienceStore
        store = ExperienceStore()

        if sub == "stats":
            mem_count = store.count_memories()
            skills = store.get_all_skills()
            insights = store.get_pending_insights()
            yield f"**Hermes Experience Stats**\n\n"
            yield f"- 记忆: {mem_count}/{settings.hermes_memory_capacity}\n"
            yield f"- Skills: {len(skills)}/{settings.hermes_skill_capacity}\n"
            yield f"- 待处理 Insight: {len(insights)}\n"
            if skills:
                yield f"\n**Registered Skills:**\n"
                for s in skills:
                    yield f"- `{s['name']}`: {s.get('description', '')[:80]}\n"
        elif sub == "memory":
            memories = store.get_memories(limit=10)
            if not memories:
                yield "暂无记忆。\n"
            for m in memories:
                yield f"- [{m.get('timestamp','')[:10]}] {m.get('intent','')[:60]} → {m.get('result','')}\n"
        elif sub == "skills":
            skills = store.get_all_skills()
            if not skills:
                yield "暂无 Skill。\n"
            for s in skills:
                steps_str = " → ".join(
                    step.get("skill_name", step.get("type", "?"))[:20]
                    for step in s.get("steps", [])
                )
                yield f"- `{s['name']}` (x{s.get('success_count',1)}): {steps_str}\n"
        elif sub == "insights":
            insights = store.get_insights(limit=10)
            if not insights:
                yield "暂无 Insight。\n"
            for i in insights:
                status = "[已应用]" if i.get("applied") else "[待处理]"
                yield f"- {status} {i.get('description','')[:80]}\n"
        elif sub == "forget":
            if "all" in args:
                import shutil
                import os
                for f in ["memory.jsonl", "skills.json", "insights.jsonl"]:
                    p = Path.home() / ".mofix" / "experience" / f
                    if p.exists():
                        os.remove(p)
                yield "所有 Hermes 经验数据已清除。\n"
        else:
            yield f"未知子命令: {sub}。可用: stats, memory, skills, insights, forget\n"

    cmd_registry.register(
        Command(
            name="experience",
            description="查询和管理 Hermes 自演进引擎的经验数据 (stats/memory/skills/insights/forget)",
            trigger_modes=[TriggerMode.USER_TRIGGER],
            handler=experience_handler,
            example="/experience stats",
        )
    )

    # ── 注册 /learn 命令 ──────────────────────────────────────────
    import uuid
    from datetime import datetime, timezone

    async def learn_handler(args: str) -> AsyncIterator[str]:
        """教导 Hermes 记住一个经验。"""
        from engine.experience.store import ExperienceStore
        store = ExperienceStore()
        memory = {
            "id": f"mem_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "intent": args[:200],
            "source": "user_teach",
            "result": "success",
        }
        store.append_memory(memory)
        yield f"[Hermes] 已记住: {args[:100]}...\n"

    cmd_registry.register(
        Command(
            name="learn",
            description="教导 Hermes 记住一个适配经验，下次相似任务时自动参考",
            trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND],
            handler=learn_handler,
            example="/learn 适配 Qwen3-8B 到 A2 需要使用 ascendc-op-develop 开发 rms_norm 算子",
        )
    )

    # 启动时初始化 MCP 管理器
    from engine.workflow.mcp_integration import init_mcp_manager, shutdown_mcp_manager
    import asyncio
    async def _init_mcp() -> None:
        if settings.mcp_auto_start:
            try:
                manager = await init_mcp_manager(settings.mcp_servers)
                tools = manager.get_all_tools()
                if tools:
                    print(f"[MCP] 已加载 {len(tools)} 个工具，来自 {len(manager.active_servers)} 台服务器: {', '.join(manager.active_servers)}")
                else:
                    print(f"[MCP] 已连接 {len(manager.active_servers)} 台服务器，但未发现工具")
            except Exception as e:
                print(f"[MCP] 初始化异常：{type(e).__name__}: {e}")

    import asyncio
    asyncio.create_task(_init_mcp())

    # Hermes 引擎初始化
    if settings.hermes_enabled:
        try:
            from engine.experience.store import ExperienceStore
            store = ExperienceStore()
            mem_count = store.count_memories()
            skill_count = len(store.get_all_skills())
            if mem_count > 0 or skill_count > 0:
                print(f"[Hermes] 已加载 {mem_count} 条记忆, {skill_count} 个可复用 Skill")
            else:
                print(f"[Hermes] 引擎就绪（尚无经验数据）")
        except Exception as e:
            print(f"[Hermes] 初始化异常: {e}")

    # 启动时异步检查 doc-agent 环境（后台任务，不阻塞启动）
    from engine.doc_tools import resolve_doc_agent_root, check_doc_agent_environment

    async def _check_doc_env() -> None:
        try:
            root = resolve_doc_agent_root()
            issues = await asyncio.to_thread(check_doc_agent_environment, root)
            if issues:
                print(f"[doc-agent] 环境检查未通过：{' / '.join(issues)}")
            else:
                print(f"[doc-agent] 环境就绪：{root}")
        except Exception as e:
            print(f"[doc-agent] 启动检查异常：{type(e).__name__}: {e}")

    asyncio.create_task(_check_doc_env())

    # 启动 Claude Code 历史记录监控上传（后台任务，不阻塞启动）
    from server.services.claude_history_uploader import HistoryWatcher

    _history_watcher: HistoryWatcher | None = None

    async def _start_history_watcher() -> None:
        global _history_watcher
        if not settings.claude_history_enabled:
            return
        try:
            _history_watcher = HistoryWatcher()
            asyncio.create_task(_history_watcher.start())
        except Exception as e:
            print(f"[claude-history] 启动监控异常：{type(e).__name__}: {e}")

    asyncio.create_task(_start_history_watcher())

    yield

    # 停止历史记录监控
    if _history_watcher:
        await _history_watcher.stop()

    # 关闭 MCP 连接
    await shutdown_mcp_manager()
    print("[MCP] 管理器已关闭")

    # 关闭时刷日志
    from lts_logger import lts_logger
    await lts_logger.shutdown()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 异常处理
app.add_exception_handler(MoFixException, mofix_exception_handler)

# 路由
app.include_router(chat.router)
app.include_router(config.router)
app.include_router(system.router)
app.include_router(eval_routes.router)
