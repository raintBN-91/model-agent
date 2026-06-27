from __future__ import annotations
"""MCP (Model Context Protocol) 集成模块。

管理到外部 MCP Server（CANNBot、ms-agent 等）的连接、工具发现和工具调用。
支持 stdio、streamable_http、sse 三种传输协议。

使用方式:
    from engine.workflow.mcp_integration import init_mcp_manager, get_mcp_manager

    await init_mcp_manager(settings.mcp_servers)
    manager = get_mcp_manager()
    result = await manager.call_tool("cannbot", "ascendc-op-develop", {"prompt": "..."})
"""


import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── 配置 ──────────────────────────────────────────────────────────────

@dataclass
class MCPConnectionConfig:
    """MCP 服务器连接配置。"""
    name: str                           # 服务器名称: "cannbot", "ms_agent"
    transport: str                      # "stdio" | "streamable_http" | "sse"
    command: str | None = None          # stdio: 启动命令
    args: list[str] | None = None       # stdio: 命令参数
    url: str | None = None              # streamable_http/sse: 服务器 URL
    enabled: bool = True                # 是否在启动时连接
    timeout: int = 60                   # 工具调用超时（秒）
    env: dict[str, str] | None = None   # 额外环境变量


# ── JSON-RPC 2.0 消息 ─────────────────────────────────────────────────

def _make_request(method: str, params: dict | None = None, req_id: int | None = None) -> dict:
    if req_id is None:
        req_id = int(time.time() * 1000) % 100000
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return msg


def _make_success_response(req_id: int, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _make_error_response(req_id: int, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


# ── 传输层抽象（Strategy 模式） ────────────────────────────────────────

class BaseTransport:
    """MCP 传输层基类。"""

    async def connect(self) -> None:
        raise NotImplementedError

    async def send(self, message: dict) -> None:
        raise NotImplementedError

    async def receive(self) -> dict:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        return False


class StdioTransport(BaseTransport):
    """通过子进程 stdin/stdout 通信的 MCP 传输层。"""

    def __init__(self, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None):
        self._command = command
        self._args = args or []
        self._env = env
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._buffer = b""

    async def connect(self) -> None:
        full_cmd = [self._command] + self._args
        logger.info(f"[MCP] 启动 stdio 子进程: {' '.join(full_cmd)}")

        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
        )

        # 读取 stderr 的后台任务（仅日志）
        stderr_task = asyncio.create_task(self._log_stderr())
        self._stderr_task = stderr_task

        self._writer = self._process.stdin
        self._reader = self._process.stdout
        self._connected = True
        logger.info(f"[MCP] stdio 子进程已启动 (PID={self._process.pid})")

    async def _log_stderr(self) -> None:
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug(f"[MCP stderr] {text}")
        except Exception:
            pass

    async def send(self, message: dict) -> None:
        if not self._writer:
            raise RuntimeError("Transport not connected")
        raw = json.dumps(message, ensure_ascii=False) + "\n"
        self._writer.write(raw.encode("utf-8"))
        await self._writer.drain()

    async def receive(self) -> dict:
        if not self._reader:
            raise RuntimeError("Transport not connected")
        while b"\n" not in self._buffer:
            chunk = await self._reader.read(65536)
            if not chunk:
                raise ConnectionError("stdio 子进程已关闭连接")
            self._buffer += chunk

        line, self._buffer = self._buffer.split(b"\n", 1)
        raw = line.decode("utf-8").strip()
        if not raw:
            return await self.receive()  # 跳过空行
        return json.loads(raw)

    async def close(self) -> None:
        self._connected = False
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            logger.info(f"[MCP] stdio 子进程已终止 (PID={self._process.pid})")
        if hasattr(self, "_stderr_task"):
            self._stderr_task.cancel()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._process is not None and self._process.returncode is None


class StreamableHttpTransport(BaseTransport):
    """通过 HTTP POST 通信的 MCP 传输层（streamable_http）。"""

    def __init__(self, url: str, timeout: int = 60):
        self._url = url
        self._timeout = timeout
        self._session = None
        self._connected = False

    async def connect(self) -> None:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("streamable_http 传输需要 httpx: pip install httpx")
        self._session = httpx.AsyncClient(timeout=self._timeout)
        # 验证连接
        try:
            resp = await self._session.head(self._url, timeout=10)
            logger.info(f"[MCP] HTTP 连接可用: {self._url} (status={resp.status_code})")
        except Exception:
            logger.warning(f"[MCP] HTTP 连接检查失败，将尝试发送请求: {self._url}")
        self._connected = True

    async def send(self, message: dict) -> None:
        if not self._session:
            raise RuntimeError("Transport not connected")
        raw = json.dumps(message, ensure_ascii=False)
        resp = await self._session.post(
            self._url,
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()

    async def receive(self) -> dict:
        # streamable_http 的响应在 send 中同步返回
        # 这里通过 _last_response 获取，需要重新设计
        raise NotImplementedError("streamable_http 请使用 send_and_receive")

    async def send_and_receive(self, message: dict) -> dict:
        """发送请求并等待响应（streamable_http 模式）。"""
        if not self._session:
            raise RuntimeError("Transport not connected")
        raw = json.dumps(message, ensure_ascii=False)
        try:
            resp = await self._session.post(
                self._url,
                content=raw,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            if resp.content:
                return resp.json()
            return _make_success_response(message.get("id", 0), None)
        except httpx.TimeoutException:
            raise TimeoutError(f"MCP 工具调用超时 ({self._timeout}s): {self._url}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"MCP HTTP 错误: {e.response.status_code} {e.response.text[:200]}")

    async def close(self) -> None:
        self._connected = False
        if self._session:
            await self._session.aclose()
            self._session = None

    @property
    def is_connected(self) -> bool:
        return self._connected


class SseTransport(BaseTransport):
    """通过 SSE (Server-Sent Events) 通信的 MCP 传输层。"""

    def __init__(self, url: str, timeout: int = 60):
        self._url = url
        self._timeout = timeout
        self._session = None
        self._connected = False
        self._response_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._receive_task: asyncio.Task | None = None

    async def connect(self) -> None:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("SSE 传输需要 httpx: pip install httpx")
        self._session = httpx.AsyncClient(timeout=None)
        # SSE 长连接
        resp = await self._session.get(self._url)
        resp.raise_for_status()
        self._connected = True

        # 后台读取 SSE 事件
        async def _read_sse():
            buffer = ""
            async for line in resp.aiter_lines():
                buffer += line + "\n"
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        await self._response_queue.put(data)
                    except json.JSONDecodeError:
                        pass

        self._receive_task = asyncio.create_task(_read_sse())

    async def send(self, message: dict) -> None:
        if not self._session:
            raise RuntimeError("Transport not connected")
        raw = json.dumps(message, ensure_ascii=False)
        resp = await self._session.post(
            self._url,
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()

    async def receive(self) -> dict:
        try:
            return await asyncio.wait_for(self._response_queue.get(), timeout=self._timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"MCP SSE 响应超时 ({self._timeout}s)")

    async def close(self) -> None:
        self._connected = False
        if self._receive_task:
            self._receive_task.cancel()
        if self._session:
            await self._session.aclose()
            self._session = None

    @property
    def is_connected(self) -> bool:
        return self._connected


def _create_transport(config: MCPConnectionConfig) -> BaseTransport:
    """根据配置创建对应的传输层实例。"""
    if config.transport == "stdio":
        if not config.command:
            raise ValueError("stdio 传输需要指定 command")
        return StdioTransport(config.command, config.args, config.env)
    elif config.transport == "streamable_http":
        if not config.url:
            raise ValueError("streamable_http 传输需要指定 url")
        return StreamableHttpTransport(config.url, config.timeout)
    elif config.transport == "sse":
        if not config.url:
            raise ValueError("sse 传输需要指定 url")
        return SseTransport(config.url, config.timeout)
    else:
        raise ValueError(f"不支持的传输协议: {config.transport}")


# ── 连接状态 ──────────────────────────────────────────────────────────

@dataclass
class MCPConnectionState:
    """单个 MCP 服务器连接状态。"""
    config: MCPConnectionConfig
    transport: BaseTransport
    tools: list[dict] = field(default_factory=list)  # 从服务器发现到的工具列表
    last_health_check: float = 0.0
    error: str | None = None


# ── MCPToolManager ───────────────────────────────────────────────────

class MCPToolManager:
    """MCP 工具管理器 — 连接注册、工具发现、工具调用。

    管理到多个 MCP Server 的生命周期，支持：
    - stdio 子进程管理（自动启动/停止）
    - streamable_http 连接
    - 自动工具发现（tools/list）
    - 工具调用路由
    - 运行时新增服务器
    """

    def __init__(self):
        self._servers: dict[str, MCPConnectionState] = {}
        self._initialized = False
        self._tool_counter = 0  # 用于生成唯一 request ID

    async def initialize(self, configs: list[MCPConnectionConfig | dict]) -> None:
        """初始化所有启用的 MCP 服务器连接。"""
        for cfg in configs:
            if isinstance(cfg, dict):
                cfg = MCPConnectionConfig(**cfg)
            if not cfg.enabled:
                logger.info(f"[MCP] 跳过已禁用的服务器: {cfg.name}")
                continue
            try:
                await self.register_server(cfg)
            except Exception as e:
                logger.error(f"[MCP] 初始化服务器 {cfg.name} 失败: {e}")
                self._servers[cfg.name] = MCPConnectionState(
                    config=cfg,
                    transport=None,
                    tools=[],
                    error=str(e),
                )
        self._initialized = True
        logger.info(f"[MCP] 初始化完成: {len(self._servers)} 台服务器")

    async def register_server(self, config: MCPConnectionConfig) -> list[dict]:
        """运行时注册一个新的 MCP 服务器。（用户可扩展性）

        流程:
        1. 创建传输层
        2. 连接服务器
        3. 调用 tools/list 发现工具
        4. 注册到 _servers

        Args:
            config: MCP 服务器连接配置

        Returns:
            服务器上可用的工具列表
        """
        if config.name in self._servers:
            logger.warning(f"[MCP] 服务器 {config.name} 已存在，将替换")
            await self._disconnect_server(config.name)

        transport = _create_transport(config)
        await transport.connect()

        # tools/list 发现
        tools = await self._discover_tools(transport, config.name)

        state = MCPConnectionState(
            config=config,
            transport=transport,
            tools=tools,
        )
        self._servers[config.name] = state
        logger.info(f"[MCP] 服务器 {config.name} 注册成功，发现 {len(tools)} 个工具")
        return tools

    async def _discover_tools(self, transport: BaseTransport,
                              server_name: str) -> list[dict]:
        """通过 tools/list 获取 MCP 服务器上的工具列表。"""
        request = _make_request("tools/list", req_id=0)
        try:
            if isinstance(transport, StreamableHttpTransport):
                response = await asyncio.wait_for(
                    transport.send_and_receive(request),
                    timeout=10,  # 10s discovery timeout
                )
            else:
                await transport.send(request)
                response = await transport.receive()

            if "error" in response:
                logger.warning(f"[MCP] {server_name} tools/list 返回错误: {response['error']}")
                return []

            tools_raw = response.get("result", {}).get("tools", [])
            for t in tools_raw:
                t["_server"] = server_name
            return tools_raw

        except Exception as e:
            logger.warning(f"[MCP] {server_name} 工具发现失败: {e}")
            return []

    def get_all_tools(self) -> list[dict]:
        """获取所有已注册服务器上的工具列表。

        返回: [{name, description, input_schema, server}, ...]
        """
        all_tools = []
        for name, state in self._servers.items():
            if state.error:
                continue
            for t in state.tools:
                all_tools.append({
                    "name": t.get("name", "unknown"),
                    "description": t.get("description", ""),
                    "input_schema": t.get("inputSchema", {}),
                    "server": name,
                })
        return all_tools

    def _next_id(self) -> int:
        self._tool_counter += 1
        return self._tool_counter

    async def call_tool(self, server: str, tool: str,
                        args: dict[str, Any]) -> str:
        """调用一个 MCP 工具。

        Args:
            server: MCP 服务器名称
            tool: 工具名称
            args: 工具参数

        Returns:
            工具的返回结果（文本格式）
        """
        state = self._servers.get(server)
        if not state:
            raise ValueError(f"MCP 服务器 '{server}' 未注册")
        if state.error:
            raise RuntimeError(f"MCP 服务器 '{server}' 不可用: {state.error}")
        if not state.transport or not state.transport.is_connected:
            raise ConnectionError(f"MCP 服务器 '{server}' 未连接")

        transport = state.transport
        req_id = self._next_id()

        # 构建 tools/call 请求
        request = _make_request("tools/call", {
            "name": tool,
            "arguments": args,
        }, req_id=req_id)

        logger.info(f"[MCP] 调用 {server}.{tool} (id={req_id})")

        try:
            if isinstance(transport, StreamableHttpTransport):
                response = await asyncio.wait_for(
                    transport.send_and_receive(request),
                    timeout=state.config.timeout,
                )
            else:
                await asyncio.wait_for(
                    transport.send(request),
                    timeout=state.config.timeout,
                )
                response = await asyncio.wait_for(
                    transport.receive(),
                    timeout=state.config.timeout,
                )
        except asyncio.TimeoutError:
            raise TimeoutError(f"MCP 工具 {server}.{tool} 调用超时 ({state.config.timeout}s)")
        except Exception as e:
            logger.error(f"[MCP] {server}.{tool} 调用失败: {e}")
            raise

        if "error" in response:
            error_info = response["error"]
            raise RuntimeError(f"MCP 工具 {server}.{tool} 返回错误: "
                               f"{error_info.get('message', 'unknown')} "
                               f"(code={error_info.get('code', -1)})")

        result = response.get("result", {})
        return self._format_mcp_result(result)

    def _format_mcp_result(self, result: dict) -> str:
        """格式化 MCP 工具返回结果（支持 TextContent/ImageContent 等）。"""
        content = result.get("content", [])
        parts = []
        for item in content:
            item_type = item.get("type", "text")
            if item_type == "text":
                parts.append(item.get("text", ""))
            elif item_type == "resource":
                resource = item.get("resource", {})
                blob = resource.get("blob", "")
                text = resource.get("text", "")
                parts.append(text or f"[资源: {resource.get('uri', 'unknown')} ({len(blob)} bytes)]")
            else:
                parts.append(f"[{item_type} 内容]")
        is_error = result.get("isError", False)
        result_text = "\n".join(parts)
        if is_error:
            result_text = f"[MCP 错误] {result_text}"
        return result_text

    async def health_check(self, server: str) -> bool:
        """检查服务器是否在线。"""
        state = self._servers.get(server)
        if not state or state.error:
            return False
        if not state.transport:
            return False
        if isinstance(state.transport, StreamableHttpTransport):
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.head(state.config.url, timeout=5)
                return resp.status_code < 500
            except Exception:
                return False
        return state.transport.is_connected

    async def _disconnect_server(self, name: str) -> None:
        """断开并移除一个服务器。"""
        state = self._servers.get(name)
        if state and state.transport:
            try:
                await state.transport.close()
            except Exception as e:
                logger.warning(f"[MCP] 关闭 {name} 连接时出错: {e}")
        self._servers.pop(name, None)

    async def shutdown(self) -> None:
        """关闭所有 MCP 连接，停止所有子进程。"""
        logger.info(f"[MCP] 正在关闭 {len(self._servers)} 台服务器...")
        for name in list(self._servers.keys()):
            await self._disconnect_server(name)
        self._initialized = False
        logger.info("[MCP] 全部关闭完成")

    @property
    def active_servers(self) -> list[str]:
        return [n for n, s in self._servers.items() if not s.error]

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# ── 全局单例 ──────────────────────────────────────────────────────────

_manager: MCPToolManager | None = None


def get_mcp_manager() -> MCPToolManager:
    """获取全局 MCPToolManager 实例。"""
    global _manager
    if _manager is None:
        _manager = MCPToolManager()
    return _manager


async def init_mcp_manager(configs: list[MCPConnectionConfig | dict]) -> MCPToolManager:
    """初始化全局 MCP 管理器并连接所有服务器。"""
    manager = get_mcp_manager()
    await manager.initialize(configs)
    return manager


async def shutdown_mcp_manager() -> None:
    """关闭全局 MCP 管理器。"""
    global _manager
    if _manager:
        await _manager.shutdown()
        _manager = None
