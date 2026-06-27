"""ms-agent MCP Server 进程管理器（兼容层 — 已废弃）。

ms-agent 现已改用 stdio 传输模式，启动时从远程仓库克隆，不再需要进程管理器。
MSAgentRunner 保留仅为向后兼容 —— 新代码应直接使用 MCPToolManager。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MSAgentRunner:
    """ms-agent MCP Server 进程生命周期管理器（兼容层 — 已废弃）。

    当前 ms-agent 已改用启动时克隆远程仓库 + stdio 传输模式。
    此类仅提供向后兼容的 API 签名。
    """

    def __init__(
        self,
        command: str = "python",
        port: int = 8001,
        host: str = "127.0.0.1",
        health_check_retries: int = 10,
        health_check_interval: float = 2.0,
        shutdown_timeout: float = 10.0,
        env: dict[str, str] | None = None,
    ):
        self._port = port
        self._host = host
        self._started = False
        self._process: asyncio.subprocess.Process | None = None
        logger.warning(
            "[ms-agent] MSAgentRunner 已废弃，请改用 MCPToolManager + StdioTransport. "
            "(参见 config.py 中 ms_agent 的 stdio 配置)"
        )

    @property
    def mcp_url(self) -> str:
        return f"http://{self._host}:{self._port}/mcp"

    async def start(self) -> str:
        self._started = True
        logger.info("[ms-agent] (兼容模式) MCP stdio 进程由 MCPToolManager 自动管理")
        return self.mcp_url

    async def is_healthy(self) -> bool:
        return self._started

    async def stop(self) -> None:
        self._started = False
        logger.info("[ms-agent] (兼容模式) 已标记为停止")
