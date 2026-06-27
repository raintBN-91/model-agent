"""Pydantic Settings — 配置管理。"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 服务配置
    app_name: str = "MoFix Backend"
    app_version: str = "0.2.0"
    port: int = 8000
    host: str = "0.0.0.0"

    # LLM 配置（Anthropic Claude，与 Claude SDK 共用 ~/.claude/settings.json）
    # 以下环境变量可选，未配置时自动读取 Claude SDK 配置文件
    anthropic_auth_token: str = ""
    anthropic_base_url: str = ""
    anthropic_model: str = ""

    # 华为云 LTS 配置（全部从环境变量读取）
    hw_access_key_id: str = ""
    hw_secret_access_key: str = ""
    hw_region: str = ""
    hw_lts_project_id: str = ""
    hw_lts_log_group_id: str = ""
    hw_lts_log_stream_id: str = ""

    # Claude Code
    claude_allowed_tools: list[str] = ["Bash", "Read", "Edit", "Write", "Glob", "Grep"]
    # 权限模式（CLI 有效值）：auto, bypassPermissions, acceptEdits, default, dontAsk, plan
    #   auto              = 自动执行并保留后台安全检查（推荐后端使用）
    #   bypassPermissions = 完全跳过权限检查（最激进，隔离环境可用）
    #   acceptEdits       = 自动接受文件编辑，其他仍询问
    #   default           = 首次使用每种工具时询问
    #   dontAsk           = 仅执行预批准工具，Skill 等会被自动拒绝
    #   plan              = 只读分析，不执行任何工具
    claude_permission_mode: str = "auto"

    # Dynamic Workflow 配置
    workflow_llm_model: str = ""           # 专用于工作流规划的模型，为空则复用 anthropic_model
    workflow_max_steps: int = 6            # 单个工作流最大步骤数
    workflow_max_concurrency: int = 3      # 最大并行数
    workflow_checkpoint_dir: str = ""      # Checkpoint 目录，为空则用 ~/.mofix/workflows
    workflow_verify_enabled: bool = True   # 默认启用验证步骤

    # MCP Server 配置
    mcp_servers: list[dict] = [
        {
            "name": "cannbot",
            "transport": "stdio",
            "command": "python3",
            "args": ["-m", "mcp.cannbot_server"],
            "enabled": True,
            "timeout": 300,
        },
        {
            "name": "ms_agent",
            "transport": "stdio",
            "command": "python3",
            "args": ["-m", "mcp.ms_agent_server"],
            "enabled": True,
            "timeout": 300,
        },
    ]
    mcp_auto_start: bool = True

    # Hermes 自演进引擎配置
    hermes_enabled: bool = True
    hermes_data_dir: str = ""           # 为空则用 ~/.mofix/experience
    hermes_memory_capacity: int = 500   # 最大记忆条数
    hermes_skill_capacity: int = 50     # 最大 Skill 数
    hermes_insight_capacity: int = 200  # 最大 insight 数
    hermes_nudge_interval: int = 3600   # Nudge 审查最小间隔（秒）

    # Claude Code 历史记录监控配置
    claude_history_enabled: bool = True      # 是否启用历史记录监控
    claude_history_project_dir: str = ""     # 显式指定 ~/.claude/projects 下项目目录；为空则自动推导

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# 全局单例
settings = Settings()
