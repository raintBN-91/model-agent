from __future__ import annotations
"""MemorySaver 封装 — 避免循环导入。"""


from typing import Any

from langgraph.checkpoint.memory import MemorySaver

# 全局共享的内存检查点（进程内共享，重启丢失）
_memory_saver = MemorySaver()


def get_memory_saver() -> MemorySaver:
    """返回全局 MemorySaver 实例，供外部模块读取对话历史。"""
    return _memory_saver


def get_thread_messages(thread_id: str | None) -> list[Any]:
    """根据 thread_id 从 MemorySaver 读取当前会话的消息历史。"""
    if not thread_id:
        return []
    try:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = _memory_saver.get_tuple(config)
        if checkpoint_tuple is None:
            return []
        checkpoint = checkpoint_tuple.checkpoint
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        return list(messages)
    except Exception:
        return []
