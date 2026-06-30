"""工具注册中心。"""

from __future__ import annotations

from . import (
    adapt_tools,
    claude_tools,
    deploy_tools,
    doc_tools,
    optimizer_tools,
    quantify_tools,
    search_tools,
    verify_tools,
)

_AGENT_MODULES = [
    search_tools,
    verify_tools,
    quantify_tools,
    adapt_tools,
    optimizer_tools,
    claude_tools,
    doc_tools,
    deploy_tools,
]


def get_all_tools() -> list:
    tools: list = []
    for mod in _AGENT_MODULES:
        tools.extend(mod.get_tools())
    return tools


def get_tool_names() -> list[str]:
    return [t.name for t in get_all_tools()]
