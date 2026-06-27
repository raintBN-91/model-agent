"""外部 Agent / MCP 归档目录。

本目录归档了 mofix 集成的外部 Agent 和 MCP 服务器的静态引用，
不再本地持久化仓库内容 — 所有远程仓库在 mofix 启动时克隆到临时目录，
MCP 化后提供服务，关闭时自动清理。

结构:
    cannbot/          CANNBot（来自 https://gitcode.com/cann/cannbot-skills）
                      启动时克隆 → 解析 SKILL.md → 注册为 MCP tools
    model-agent/      Model-Agent（来自 https://gitcode.com/Ascend/model-agent）
                      启动时克隆 → 解析 AGENT 配置 → 注册为 MCP tools
    ms_agent/         msAgent（来自 https://gitcode.com/hewenbo/msagent）
                      启动时克隆 → 解析 Agent 能力 → 注册为 MCP tools
"""
