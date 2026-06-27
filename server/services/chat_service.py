from __future__ import annotations
"""对话服务 — OpenAI 格式 ↔ LangChain 转换 + 命令分发。"""


import asyncio
import json
import logging
import time
import uuid
from typing import AsyncIterator

logger = logging.getLogger(__name__)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from server.config import settings
from engine.agent_engine import build_agent
from engine.exceptions import PendingConfirmationError
from engine.llm_factory import extract_text_content, get_llm_model_name, handle_api_error
from server.models.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatMessage,
    Choice,
    DeltaContent,
)
from engine.claude_tools import _chunk_text
from engine.doc_tools import DOC_SUBCOMMANDS
from engine.workflow import DynamicPlanner, WorkflowExecutor
from engine.workflow.brainstorming import (
    BrainstormingSession,
    BrainstormQuestion,
    BrainstormState,
)
from commands.registry import TriggerMode, cmd_registry
from lts_logger import lts_logger

# 全局内存表：c_id → pending tool_calls（等待用户确认）
_pending_tool_calls: dict[str, list[dict]] = {}

# 全局内存表：c_id → 活跃的 BrainstormingSession
_brainstorm_sessions: dict[str, dict] = {}


def _build_sse_chunk(
    chunk_id: str,
    model: str,
    delta: DeltaContent,
    finish_reason: str | None = None,
    requires_confirmation: bool | None = None,
    requires_input: bool | None = None,
    brainstorm_state: str | None = None,
) -> str:
    """构建 SSE data: 行。"""
    obj = ChatCompletionChunk(
        id=chunk_id,
        created=int(time.time()),
        model=model,
        choices=[Choice(
            delta=delta,
            finish_reason=finish_reason,
            requires_confirmation=requires_confirmation,
            requires_input=requires_input,
            brainstorm_state=brainstorm_state,
        )],
    )
    return f"data: {obj.model_dump_json(exclude_none=True)}\n\n"


def _convert_messages(msgs: list[ChatMessage]) -> list[BaseMessage]:
    """OpenAI 格式 → LangChain BaseMessage。"""
    result: list[BaseMessage] = []
    for m in msgs:
        if m.role == "system":
            result.append(SystemMessage(content=m.content or ""))
        elif m.role == "user":
            result.append(HumanMessage(content=m.content or ""))
        elif m.role == "assistant":
            result.append(AIMessage(content=m.content or ""))
        elif m.role == "tool":
            result.append(ToolMessage(content=m.content or "", name=m.name or "", tool_call_id=m.tool_call_id or ""))
    return result


async def _execute_command(cmd_name: str, args: str) -> AsyncIterator[str]:
    """执行用户主动触发的命令，产出 OpenAI SSE 流。"""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model = get_llm_model_name()

    cmd = cmd_registry.get(cmd_name)
    if cmd is None:
        logger.warning("Unknown command: cmd=%s, args=%.80s", cmd_name, args)
        yield _build_sse_chunk(
            chunk_id, model,
            DeltaContent(role="assistant", content=f"未知命令 `/{cmd_name}`。"),
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
        return

    logger.info("Executing command: cmd=%s, args=%.120s", cmd_name, args)

    if TriggerMode.USER_TRIGGER not in cmd.trigger_modes:
        yield _build_sse_chunk(
            chunk_id, model,
            DeltaContent(role="assistant", content=f"命令 `/{cmd_name}` 不支持用户直接触发。"),
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
        return

    yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=""))
    yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=f"正在执行 /{cmd_name}...\n\n"))

    start = time.time()
    try:
        raw = cmd.handler(args)
        if hasattr(raw, "__aiter__"):
            # handler 返回异步生成器 → 流式逐条输出
            async for chunk in raw:
                yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=chunk))
        else:
            # handler 返回协程 → 等待完成后一次性输出
            result = await raw
            yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=result))
        lts_logger.log(
            "tool_invoke",
            tool_name=cmd_name,
            duration_ms=int((time.time() - start) * 1000),
            status="success",
        )
    except Exception as e:
        lts_logger.log(
            "tool_invoke",
            tool_name=cmd_name,
            duration_ms=int((time.time() - start) * 1000),
            status="error",
            error=str(e),
        )
        yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=json.dumps({
            "error": {"message": str(e), "type": "command_error", "code": "execution_failed"},
        }, ensure_ascii=False)))

    yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=""), finish_reason="stop")
    yield "data: [DONE]\n\n"


def _extract_stream_config(request: ChatCompletionRequest) -> tuple[dict | None, dict]:
    """构建 LangGraph stream 配置。"""
    stream_config = {"configurable": {"thread_id": request.c_id or f"default-{id(request)}"}}

    stream_kwargs: dict = {"stream_mode": ["messages", "values"]}
    stream_kwargs["config"] = stream_config
    return stream_config, stream_kwargs


def _yield_agent_chunks(
    agent,
    chunk_id: str,
    model: str,
    stream_input,
    stream_kwargs: dict,
) -> AsyncIterator[str]:
    """通用 Agent 流式执行生成器，产出 SSE chunks。"""
    for mode, payload in agent.stream(stream_input, **stream_kwargs):
        if mode == "messages":
            if not isinstance(payload, tuple) or not payload:
                continue
            msg = payload[0]

            if isinstance(msg, AIMessage):
                text = extract_text_content(msg.content)
                if text:
                    yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=text))

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield _build_sse_chunk(
                            chunk_id, model,
                            DeltaContent(
                                role="assistant",
                                content="",
                                tool_calls=[{
                                    "index": 0,
                                    "id": tc.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("name", ""),
                                        "arguments": str(tc.get("args", {})),
                                    },
                                }]
                            ),
                        )

            elif isinstance(msg, ToolMessage):
                content = str(msg.content)
                if getattr(msg, "name", None) == "invoke_claude_skill":
                    for chunk in _chunk_text(content):
                        yield _build_sse_chunk(
                            chunk_id, model,
                            DeltaContent(role="assistant", content=chunk),
                        )
                else:
                    yield _build_sse_chunk(
                        chunk_id, model,
                        DeltaContent(
                            role="assistant",
                            content=content[:2000],
                        ),
                    )


async def _finish_stream(
    chunk_id: str,
    model: str,
    request_id: str,
    start_time: float,
    status: str = "success",
) -> AsyncIterator[str]:
    """发送 SSE 结束标记和 LTS 日志。"""
    yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=""), finish_reason="stop")
    yield "data: [DONE]\n\n"
    lts_logger.log(
        "chat_completion_end",
        request_id=request_id,
        model=model,
        duration_ms=int((time.time() - start_time) * 1000),
        status=status,
    )


async def with_heartbeat(
    source: AsyncIterator[str],
    interval: float = 5.0,
) -> AsyncIterator[str]:
    """为 SSE 异步生成器注入心跳包，防止长时间无数据导致链路断开。"""
    sentinel = object()
    queue: asyncio.Queue[str | object] = asyncio.Queue()

    async def _forward() -> None:
        async for chunk in source:
            await queue.put(chunk)
        await queue.put(sentinel)

    task = asyncio.create_task(_forward())
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                yield "data: [PING]\n\n"
                continue

            if chunk is sentinel:
                break
            yield chunk
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ── Brainstorming 辅助函数 ─────────────────────────────────────────

async def _send_brainstorm_question(
    chunk_id: str,
    model: str,
    question: BrainstormQuestion,
) -> AsyncIterator[str]:
    """发送脑暴问题给用户（requires_input=true）。"""
    content = f"## {question.question}\n\n"
    if question.options:
        for i, opt in enumerate(question.options):
            content += f"- **{i}**. {opt}\n"
    if question.allow_custom:
        content += "\n*也可以输入自定义答案。*"
    content += "\n\n请回复选择对应的选项编号或输入自定义内容。"

    yield _build_sse_chunk(
        chunk_id, model,
        DeltaContent(role="assistant", content=content),
        requires_input=True,
        brainstorm_state=BrainstormState.CLARIFYING.value,
    )


async def _send_brainstorm_proposals(
    chunk_id: str,
    model: str,
    proposals: list[dict],
) -> AsyncIterator[str]:
    """发送方案列表给用户选择。"""
    content = "## 基于已收集的信息，我为您设计了以下方案：\n\n"
    for i, p in enumerate(proposals):
        content += f"### {i}. {p.get('title', f'方案 {i}')}\n"
        content += f"{p.get('description', '')}\n\n"
        pros = p.get("pros", [])
        if pros:
            content += "**优点**: " + ", ".join(f"[+] {pr}" for pr in pros) + "\n"
        cons = p.get("cons", [])
        if cons:
            content += "**缺点**: " + ", ".join(f"[-] {co}" for co in cons) + "\n"
        effort = p.get("estimated_effort", "")
        if effort:
            content += f"**预估工作量**: {effort}\n"
        content += "\n"
    content += "请选择方案编号（0-{}），或输入 reject 重新讨论。".format(len(proposals) - 1)

    yield _build_sse_chunk(
        chunk_id, model,
        DeltaContent(role="assistant", content=content),
        requires_input=True,
        brainstorm_state=BrainstormState.PROPOSING.value,
    )


async def _send_brainstorm_design_doc(
    chunk_id: str,
    model: str,
    design_doc: dict,
) -> AsyncIterator[str]:
    """发送设计文档让用户确认。"""
    dd = design_doc
    content = "## 设计文档\n\n"
    content += f"**目标**: {dd.get('goal', 'N/A')}\n"
    content += f"**平台**: {dd.get('platform', 'N/A')}\n"
    content += f"**模型**: {dd.get('model_name', 'N/A')}\n"
    content += f"**推理框架**: {dd.get('framework', 'N/A')}\n\n"
    content += f"**方案**: {dd.get('approach', 'N/A')}\n\n"
    wf = dd.get("workflow_plan", [])
    if wf:
        content += "**执行计划**:\n" + "\n".join(f"  {s}" for s in wf) + "\n\n"
    risks = dd.get("risks", [])
    if risks:
        content += "**风险**: " + ", ".join(f"[!] {r}" for r in risks) + "\n\n"
    content += "---\n请输入 **approve** 确认执行，或输入 **reject** 并提供原因。"

    yield _build_sse_chunk(
        chunk_id, model,
        DeltaContent(role="assistant", content=content),
        requires_input=True,
        brainstorm_state=BrainstormState.REVIEWING.value,
    )


async def _handle_brainstorm_request(
    request: ChatCompletionRequest,
    chunk_id: str,
    model: str,
) -> AsyncIterator[str]:
    """处理脑暴会话的多轮交互。

    在 chat_completions_stream 中检测到 ``c_id in _brainstorm_sessions`` 时调用。
    """
    global _brainstorm_sessions
    session_data = _brainstorm_sessions.get(request.c_id)
    if not session_data:
        return

    session = BrainstormingSession.from_dict(session_data)

    # 解析 brainstorm 回复
    brainstorm_raw = request.brainstorm or ""
    answer = brainstorm_raw.strip()

    if not answer and session.state in (BrainstormState.APPROVED, BrainstormState.COMPLETED):
        # 已批准，直接执行
        async for line in _execute_brainstorm_approved(session, chunk_id, model):
            yield line
        del _brainstorm_sessions[request.c_id]
        return

    if not answer:
        # 没有新回答但会话存在 → 重新发送当前状态
        async for line in _resend_brainstorm_state(session, chunk_id, model):
            yield line
        return

    # 处理 approve / reject 关键词
    if session.state == BrainstormState.REVIEWING:
        if answer.strip().lower() == "approve":
            result = session.approve()
            if result.get("type") == "complete":
                del _brainstorm_sessions[request.c_id]
                context = result.get("context", {})
                async for line in _execute_workflow_with_context(context, chunk_id, model):
                    yield line
            else:
                _brainstorm_sessions[request.c_id] = session.to_dict()
            return

        if answer.strip().lower().startswith("reject"):
            reason = answer[6:].strip() if len(answer) > 6 else ""
            result = session.reject(reason)
            async for line in _dispatch_brainstorm_result(result, session, chunk_id, model, request):
                yield line
            _brainstorm_sessions[request.c_id] = session.to_dict()
            return

    else:
        # 处理普通回答
        result = session.answer("", answer)
        async for line in _dispatch_brainstorm_result(result, session, chunk_id, model, request):
            yield line

        # 更新会话状态
        if result.get("type") == "complete":
            # 脑暴结束 → 执行工作流
            del _brainstorm_sessions[request.c_id]
            context = result.get("context", {})
            async for line in _execute_workflow_with_context(context, chunk_id, model):
                yield line
        else:
            _brainstorm_sessions[request.c_id] = session.to_dict()


async def _execute_brainstorm_approved(
    session: BrainstormingSession,
    chunk_id: str,
    model: str,
) -> AsyncIterator[str]:
    """脑暴已批准，直接执行工作流。"""
    yield _build_sse_chunk(chunk_id, model, DeltaContent(
        role="assistant", content="## 设计已确认，开始执行工作流...\n\n"
    ))
    context = session._build_context()
    async for line in _execute_workflow_with_context(context, chunk_id, model):
        yield line


async def _resend_brainstorm_state(
    session: BrainstormingSession,
    chunk_id: str,
    model: str,
) -> AsyncIterator[str]:
    """重新发送当前脑暴状态（用户没有提供新回答时）。"""
    if session.state == BrainstormState.CLARIFYING:
        # 找最后一个 question
        for entry in reversed(session.history):
            if entry.get("type") == "question":
                q = BrainstormQuestion(
                    question_id=entry.get("question_id", ""),
                    question=entry.get("content", ""),
                    options=[],
                )
                async for line in _send_brainstorm_question(chunk_id, model, q):
                    yield line
                return
    elif session.state == BrainstormState.PROPOSING:
        async for line in _send_brainstorm_proposals(
            chunk_id, model,
            [p.model_dump() for p in session.proposals],
        ):
            yield line
    elif session.state == BrainstormState.REVIEWING:
        async for line in _send_brainstorm_design_doc(
            chunk_id, model,
            session.design_doc.model_dump() if session.design_doc else {},
        ):
            yield line

    yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=""), finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _dispatch_brainstorm_result(
    result: dict,
    session: BrainstormingSession,
    chunk_id: str,
    model: str,
    request: ChatCompletionRequest,
) -> AsyncIterator[str]:
    """根据脑暴结果类型分发 SSE 输出。"""
    rtype = result.get("type")

    if rtype == "question":
        q = result.get("question")
        if q:
            async for line in _send_brainstorm_question(chunk_id, model, q):
                yield line

    elif rtype == "proposals":
        proposals = result.get("proposals", [])
        async for line in _send_brainstorm_proposals(chunk_id, model, proposals):
            yield line

    elif rtype == "design_doc":
        dd = result.get("design_doc", {})
        async for line in _send_brainstorm_design_doc(chunk_id, model, dd):
            yield line

    elif rtype == "complete":
        yield _build_sse_chunk(chunk_id, model, DeltaContent(
            role="assistant",
            content="## 脑暴完成，设计已确认！\n\n开始执行工作流...\n\n",
        ))

    else:
        yield _build_sse_chunk(chunk_id, model, DeltaContent(
            role="assistant", content=f"[脑暴] 未知响应类型: {rtype}",
        ))

    yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=""), finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _execute_workflow_with_context(
    context: dict,
    chunk_id: str,
    model: str,
) -> AsyncIterator[str]:
    """使用脑暴产生的 context 直接执行工作流（流式推送步骤进度）。"""
    try:
        planner = DynamicPlanner()
        user_input = context.get("user_input", "")
        dimensions = context.get("dimensions", {})
        enriched_input = user_input
        if dimensions:
            enriched_input += "\n[已确认上下文]\n" + "\n".join(f"{k}: {v}" for k, v in dimensions.items())

        plan = planner.plan(enriched_input)

        if not plan.steps:
            yield _build_sse_chunk(chunk_id, model, DeltaContent(
                role="assistant", content="未能生成工作流步骤。\n",
            ))
            return

        yield _build_sse_chunk(chunk_id, model, DeltaContent(
            role="assistant", content=f"**工作流计划**: {len(plan.steps)} 个步骤\n\n",
        ))

        yield _build_sse_chunk(chunk_id, model, DeltaContent(
            role="assistant", content="**开始执行工作流...**\n\n",
        ))

        # 使用 progress_callback + 后台任务实现流式步骤进度
        import asyncio as _asyncio
        _progress_queue: _asyncio.Queue = _asyncio.Queue()

        async def _on_progress(event: dict):
            await _progress_queue.put(event)

        executor = WorkflowExecutor()
        _exec_task = _asyncio.ensure_future(executor.execute(plan, _on_progress))

        while not _exec_task.done():
            try:
                event = await _asyncio.wait_for(_progress_queue.get(), timeout=0.5)
                evt = event["event"]
                if evt == "step_start":
                    yield _build_sse_chunk(chunk_id, model, DeltaContent(
                        role="assistant",
                        content=f"\n**[RUNNING] {event['step']}** (type: {event['type']})\n",
                    ))
                elif evt == "step_output":
                    yield _build_sse_chunk(chunk_id, model, DeltaContent(
                        role="assistant", content=event.get("chunk", ""),
                    ))
                elif evt == "step_end":
                    icon = "[PASS]" if event["status"] == "completed" else "[FAIL]"
                    yield _build_sse_chunk(chunk_id, model, DeltaContent(
                        role="assistant",
                        content=f"\n**[{icon}] {event['step']}** 完成\n",
                    ))
                    if event.get("error"):
                        yield _build_sse_chunk(chunk_id, model, DeltaContent(
                            role="assistant", content=f"  错误: {event['error']}\n",
                        ))
                elif evt == "step_retry":
                    yield _build_sse_chunk(chunk_id, model, DeltaContent(
                        role="assistant",
                        content=f"\n**[RETRY] {event['step']}** 第{event['attempt']}次重试\n",
                    ))
            except _asyncio.TimeoutError:
                # 超时也 yield 空 chunk 保持 SSE 连接活跃
                pass

        ctx = await _exec_task

        total = len(plan.steps)
        completed = sum(1 for s in plan.steps if s.status.value == "completed")
        failed = sum(1 for s in plan.steps if s.status.value == "failed")
        yield _build_sse_chunk(chunk_id, model, DeltaContent(
            role="assistant",
            content=f"\n**摘要**: {completed}/{total} 步骤完成" + (f"，{failed} 失败" if failed else "") + "\n",
        ))

    except Exception as e:
        logger.exception("Workflow execution failed")
        yield _build_sse_chunk(chunk_id, model, DeltaContent(
            role="assistant", content=f"[错误] 工作流执行失败: {type(e).__name__}: {e}\n",
        ))


async def chat_completions_stream(
    request: ChatCompletionRequest,
) -> AsyncIterator[str]:
    """主入口：OpenAI 格式请求 → OpenAI 格式 SSE 流。"""
    global _brainstorm_sessions, _pending_tool_calls
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    model = request.model or get_llm_model_name()
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # 1. 检测用户主动触发的命令
    last_msg = request.messages[-1].content or ""
    if last_msg.strip().startswith("/"):
        parts = last_msg[1:].strip().split(maxsplit=1)
        cmd_name = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        # /workflow 命令：评估模糊度，仅当核心意图不明确时才触发脑暴
        if cmd_name == "workflow" and request.c_id:
            # 如果已有活跃的脑暴会话，将本次输入作为脑暴回复处理
            if request.c_id in _brainstorm_sessions:
                request.brainstorm = args
                async for line in _handle_brainstorm_request(request, chunk_id, model):
                    yield line
                return

            from engine.workflow.brainstorming import assess_ambiguity
            score, missing_dimensions = await assess_ambiguity(args)
            if score > 0.45 and missing_dimensions:
                session = BrainstormingSession(session_id=request.c_id)
                result = session.start(args)
                question = result.get("question") if isinstance(result, dict) else result
                _brainstorm_sessions[request.c_id] = session.to_dict()

                lts_logger.log("brainstorm_start",
                    c_id=request.c_id, input=args, score=score, missing=missing_dimensions)

                async for line in _dispatch_brainstorm_result(result, session, chunk_id, model, request):
                    yield line
                return
            # 模糊度低 → 继续正常执行

        # /doc 子命令不匹配时透传给 LLM（走正常 Agent 流程）
        if cmd_name == "doc":
            sub = args.strip().split(maxsplit=1)[0].lower() if args else ""
            if sub not in DOC_SUBCOMMANDS:
                # 透传：继续走下方 Agent 流程，不做 return
                pass
            else:
                async for line in _execute_command(cmd_name, args):
                    yield line
                return
        else:
            async for line in _execute_command(cmd_name, args):
                yield line
            return

    # ── Brainstorming 辅助函数 ─────────────────────────────────────────

    # 1.5 脑暴会话检测：处理多轮 brainstorm 回复
    if request.c_id and request.c_id in _brainstorm_sessions:
        async for line in _handle_brainstorm_request(request, chunk_id, model):
            yield line
        return

    # 2. 转换 messages
    lc_messages = _convert_messages(request.messages)

    # 3. 提取用户 system prompt，避免与 Agent 的 system prompt 重复
    user_system_parts: list[str] = []
    chat_messages: list[BaseMessage] = []
    for m in lc_messages:
        if isinstance(m, SystemMessage):
            user_system_parts.append(m.content or "")
        else:
            chat_messages.append(m)
    user_system_prompt = "\n".join(user_system_parts) if user_system_parts else ""

    # 4. 构建 Agent
    agent = build_agent(
        streaming=True,
        cmd_registry=cmd_registry,
        user_system_prompt=user_system_prompt,
    )

    # 5. 构建 stream 配置
    stream_config, stream_kwargs = _extract_stream_config(request)

    # 6. LTS 记录开始
    lts_logger.log(
        "chat_completion_start",
        request_id=request_id,
        model=model,
        message_count=len(request.messages),
    )
    start_time = time.time()

    # 7. 处理 pending confirmation（恢复流程）
    global _pending_tool_calls
    if request.c_id and request.c_id in _pending_tool_calls:
        pending = _pending_tool_calls.pop(request.c_id)

        if request.confirmation == "confirmed":
            # 确认：从 checkpoint 恢复，graph 自动执行 tools → call_model
            try:
                yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=""))
                for chunk in _yield_agent_chunks(agent, chunk_id, model, None, stream_kwargs):
                    yield chunk
                async for chunk in _finish_stream(chunk_id, model, request_id, start_time):
                    yield chunk
            except Exception as e:
                handle_api_error(e)
                lts_logger.log("llm_error", request_id=request_id, error_type=type(e).__name__, detail=str(e))
                raise
            return

        elif request.confirmation == "rejected":
            # 拒绝：注入拒绝消息到 checkpoint，然后恢复
            try:
                if stream_config:
                    current_state = agent.get_state(stream_config)
                    messages = list(current_state.values.get("messages", []))
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                            reject_msgs = []
                            for tc in last_msg.tool_calls:
                                reject_msgs.append(
                                    ToolMessage(
                                        content="用户拒绝了该工具调用。",
                                        tool_call_id=tc.get("id", ""),
                                        name=tc.get("name", ""),
                                    )
                                )
                            agent.update_state(
                                stream_config,
                                {"messages": messages + reject_msgs},
                                as_node="tools",
                            )
                yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=""))
                for chunk in _yield_agent_chunks(agent, chunk_id, model, None, stream_kwargs):
                    yield chunk
                async for chunk in _finish_stream(chunk_id, model, request_id, start_time):
                    yield chunk
            except Exception as e:
                handle_api_error(e)
                lts_logger.log("llm_error", request_id=request_id, error_type=type(e).__name__, detail=str(e))
                raise
            return

        else:
            # 有 pending 但未传 confirmation
            # 不能抛异常（StreamingResponse 已启动），改为 yield 错误 SSE
            error_msg = (
                f"当前有工具调用等待确认（c_id={request.c_id}），"
                f"请发送 confirmation: confirmed 或 confirmation: rejected"
            )
            yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=error_msg))
            yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=""), finish_reason="stop")
            yield "data: [DONE]\n\n"
            lts_logger.log(
                "chat_completion_end",
                request_id=request_id,
                model=model,
                duration_ms=int((time.time() - start_time) * 1000),
                status="error",
                error=error_msg,
            )
            return
            

    # 8. 正常流式执行 Agent
    try:
        yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=""))

        pending_tool_calls: list[dict] | None = None
        pending_intent_text: str = ""  # 缓存 Agent 决定调用工具前的解释文本

        for mode, payload in agent.stream({"messages": chat_messages}, **stream_kwargs):
            if mode == "messages":
                if not isinstance(payload, tuple) or not payload:
                    continue
                msg = payload[0]

                if isinstance(msg, AIMessage):
                    text = extract_text_content(msg.content)
                    if text:
                        yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=text))
                        pending_intent_text = text  # 缓存意图文本

                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield _build_sse_chunk(
                                chunk_id, model,
                                DeltaContent(
                                    role="assistant",
                                    content="",
                                    tool_calls=[{
                                        "index": 0,
                                        "id": tc.get("id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": tc.get("name", ""),
                                            "arguments": str(tc.get("args", {})),
                                        },
                                    }]
                                ),
                            )
                        # 需要确认拦截：有 c_id 时保存 pending 并 break
                        if request.c_id:
                            pending_tool_calls = msg.tool_calls
                            break

                elif isinstance(msg, ToolMessage):
                    content = str(msg.content)
                    if getattr(msg, "name", None) == "invoke_claude_skill":
                        for chunk in _chunk_text(content):
                            yield _build_sse_chunk(
                                chunk_id, model,
                                DeltaContent(role="assistant", content=chunk),
                            )
                    else:
                        yield _build_sse_chunk(
                            chunk_id, model,
                            DeltaContent(
                                role="assistant",
                                content=content[:2000],
                            ),
                        )

        # 检查是否因需要确认而 break
        if pending_tool_calls:
            _pending_tool_calls[request.c_id] = pending_tool_calls
            tool_names = [tc.get("name", "unknown") for tc in pending_tool_calls]

            if pending_intent_text:
                confirm_text = (
                    f"{pending_intent_text}（即将调用 {', '.join(tool_names)}）"
                    f"是否确认执行？"
                )
            else:
                confirm_text = (
                    f"Agent 即将调用工具：{', '.join(tool_names)}，是否确认执行？"
                    if len(tool_names) == 1
                    else f"Agent 即将调用以下工具：{', '.join(tool_names)}，是否确认执行？"
                )

            yield _build_sse_chunk(
                chunk_id, model,
                DeltaContent(role="assistant", content=confirm_text),
                requires_confirmation=True,
            )
            yield _build_sse_chunk(chunk_id, model, DeltaContent(role="assistant", content=""), finish_reason="stop")
            yield "data: [DONE]\n\n"
            lts_logger.log(
                "chat_completion_end",
                request_id=request_id,
                model=model,
                duration_ms=int((time.time() - start_time) * 1000),
                status="success",
            )
            return

        # 正常结束
        async for chunk in _finish_stream(chunk_id, model, request_id, start_time):
            yield chunk

    except Exception as e:
        handle_api_error(e)
        lts_logger.log(
            "llm_error",
            request_id=request_id,
            error_type=type(e).__name__,
            detail=str(e),
            duration_ms=int((time.time() - start_time) * 1000),
        )
        raise
