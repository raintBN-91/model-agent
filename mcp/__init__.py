"""MCP Server 包装器 — 外部 Agent 的 MCP 服务端实现。

所有服务器均在启动时从远程仓库克隆，不本地持久化。

当前包含:
- cannbot_server: CANNBot skills → MCP tools (启动时从 gitcode.com/cann/cannbot-skills 克隆)
- ms_agent_server: msAgent → MCP tools (启动时从 gitcode.com/hewenbo/msagent 克隆)
- ms_agent_runner: 旧版兼容层（已废弃）
"""
