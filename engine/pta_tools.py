"""PTA Agent 调用工具 — 支持基于 thread_id 的多轮对话记忆。"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path

# 默认从与 mofix-new 同目录的 pta-agent 中加载（如 search-agent）
_PTA_AGENT_ROOT = Path(__file__).resolve().parents[1] / "pta-agent"

# PTA 私有记忆缓存：thread_id -> [(user_prompt, assistant_response), ...]
# 由于 PTA Agent 本身无记忆功能，由 mofix 在多次 /pta 调用之间维护上下文。
# 生命周期与主 Agent 的 MemorySaver 一致：进程重启后清空。
_pta_thread_memory: dict[str, list[tuple[str, str]]] = {}

def _ensure_pta_in_path() -> None:
    if _PTA_AGENT_ROOT.exists() and str(_PTA_AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PTA_AGENT_ROOT))


def _build_pta_context(prompt: str, thread_id: str | None) -> str:
    """为本次 PTA 查询组装历史上下文。
    合并两个来源：
    主 Agent MemorySaver 中的普通聊天历史；
    本模块缓存的历次 /pta 问答。
    最后把当前 prompt 接在上下文后面返回。
    """
    from engine.memory import get_thread_messages
    from engine.claude_tools import _format_history_as_context

    parts: list[str] = []

    #  主会话历史（普通聊天）
    main_context = _format_history_as_context(get_thread_messages(thread_id))
    if main_context:
        # 去掉 _format_history_as_context 自带的结尾分隔线，统一在最后加
        inner = main_context
        prefix = "以下是当前会话的上下文供你参考：\n\n"
        suffix = "\n\n---\n\n"
        if inner.startswith(prefix):
            inner = inner[len(prefix) :]
        if inner.endswith(suffix):
            inner = inner[: -len(suffix)]
        if inner.strip():
            parts.append(inner.strip())

    #  历次 /pta 问答（PTA Agent 自身不会保存，需要 mofix 自己维护）
    if thread_id:
        for user_prompt, assistant_response in _pta_thread_memory.get(thread_id, []):
            user_text = user_prompt[:800] + "...[截断]" if len(user_prompt) > 800 else user_prompt
            assistant_text = (
                assistant_response[:1200] + "...[截断]"
                if len(assistant_response) > 1200
                else assistant_response
            )
            parts.append(f"【用户】{user_text}\n【助手】{assistant_text}")

    if not parts:
        return prompt

    context = "\n\n".join(parts)

    # 总长度保护，避免 prompt 爆炸
    max_chars = 4000
    if len(context) > max_chars:
        context = context[-max_chars:]
        first_nl = context.find("\n")
        if first_nl != -1:
            context = context[first_nl + 1 :]
        context = "...[前文省略]\n" + context

    return f"以下是当前会话的上下文供你参考：\n\n{context}\n\n---\n\n{prompt}"


def _save_pta_exchange(thread_id: str | None, prompt: str, response: str) -> None:
    """保存本次 /pta 问答到私有缓存，供下一次 /pta 读取。"""
    if not thread_id:
        return
    _pta_thread_memory.setdefault(thread_id, []).append((prompt, response))
    # 防止无限增长：单个线程最多保留最近 20 轮 PTA 问答
    _pta_thread_memory[thread_id] = _pta_thread_memory[thread_id][-20:]


async def stream_pta_query(prompt: str, thread_id: str | None = None) -> AsyncIterator[str]:
    """流式调用 PTA Agent，支持基于 thread_id 的多轮上下文。

    默认从与 mofix-new 同目录的 pta-agent 加载（通过 pta-agent/src 包导入）。
    """
    if not _PTA_AGENT_ROOT.exists():
        yield (
            f"[pta] 未找到 pta-agent 目录：{_PTA_AGENT_ROOT}。"
            f"请将 pta-agent 克隆到与 mofix-new 同级目录。"
        )
        return

    _ensure_pta_in_path()

    try:
        from src.api import pta
    except Exception as e:
        yield f"[pta] 加载 PTA Agent 失败：{type(e).__name__}: {e}"
        return

    # 根据 thread_id 组装历史上下文并拼接到 prompt 前。
    # PTA Agent 本身无记忆，由 mofix 把历史塞进 prompt 实现多轮对话。
    full_prompt = _build_pta_context(prompt, thread_id)

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    response_parts: list[str] = []

    def _on_text(chunk: str) -> None:
        response_parts.append(chunk)
        loop.call_soon_threadsafe(queue.put_nowait, chunk)

    def _run_pta() -> None:
        try:
            # 保持 memory=False，因为跨调用的记忆已由 mofix 维护。
            # 若 PTA Agent 后续支持 thread 级记忆，可再考虑开启。
            pta(full_prompt, memory=False, on_text=_on_text)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, f"[pta] 调用失败：{type(e).__name__}: {e}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.create_task(asyncio.to_thread(_run_pta))

    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # 流结束后保存本次问答，使下一次 /pta 能继续引用。
    _save_pta_exchange(thread_id, prompt, "".join(response_parts))
