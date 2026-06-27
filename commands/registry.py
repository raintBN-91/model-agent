"""命令注册中心 — 支持三种触发方式的命令扩展机制。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable


class TriggerMode(str, Enum):
    USER_TRIGGER = "user_trigger"
    LLM_RECOMMEND = "llm_recommend"
    LLM_AUTO = "llm_auto"


@dataclass
class Command:
    name: str
    description: str
    trigger_modes: list[TriggerMode]
    handler: Callable[..., Awaitable[str]] | Callable[..., AsyncIterator[str]]
    example: str = ""


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, cmd: Command) -> None:
        self._commands[cmd.name] = cmd

    def get(self, name: str) -> Command | None:
        return self._commands.get(name)

    def list_all(self) -> list[Command]:
        return list(self._commands.values())

    def list_by_mode(self, mode: TriggerMode) -> list[Command]:
        return [c for c in self._commands.values() if mode in c.trigger_modes]

    def build_system_prompt_addon(self) -> str:
        lines: list[str] = ["\n## 可用命令系统"]

        # LLM_AUTO 命令 → 说明可直接调用的工具
        auto_cmds = self.list_by_mode(TriggerMode.LLM_AUTO)
        if auto_cmds:
            lines.append("以下命令你可以直接调用对应工具执行：")
            for cmd in auto_cmds:
                lines.append(f"- `invoke_{cmd.name}`：{cmd.description}")

        # LLM_RECOMMEND 命令 → 说明推荐规则
        recommend_cmds = self.list_by_mode(TriggerMode.LLM_RECOMMEND)
        if recommend_cmds:
            lines.append(
                "\n当用户需求匹配以下命令时，请在回复中输出对应的 `/命令名` 提示词供用户复制："
            )
            for cmd in recommend_cmds:
                lines.append(f"- **/{cmd.name}**：{cmd.description} → 格式：`{cmd.example}`")

        return "\n".join(lines) if len(lines) > 1 else ""


# 全局注册表
cmd_registry = CommandRegistry()
