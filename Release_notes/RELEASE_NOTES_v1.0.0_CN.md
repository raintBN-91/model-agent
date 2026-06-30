# 发布说明

## 2026.06.30

本次发布是一次重大架构升级，将后端从一个简单的 OpenAI 兼容聊天代理改造为功能完备的模型优化框架，具备动态工作流编排、自演进经验引擎和 MCP（模型上下文协议）集成能力。

### 核心亮点

- **动态工作流引擎**：以基于 DAG 的 `DynamicPlanner` + `WorkflowExecutor` 替代原有的静态意图解析器，支持 LLM 生成计划、并行步骤执行、重试逻辑、检查点恢复及实时 SSE 进度推送。
- **Hermes 自演进引擎**：新增经验子系统（`app/tools/experience/`），持久化适配记忆、技能和洞察，使框架能够从历史执行中学习并主动建议优化方案。
- **MCP 集成**：支持模型上下文协议，内置 Cannbot 和 MS Agent 两个 MCP Server，提供生命周期管理及工作流引擎集成。
- **PTA Agent**：新增专用 PyTorch-Ascend Agent（`/pta`），支持基于 `thread_id` 的多轮对话记忆，专为 `torch_npu` 开发场景设计。
- **Anthropic Claude 原生接入**：将 LLM 提供商从 MiniMax/MoonShot（OpenAI 兼容）切换为 Anthropic Claude，并通过新增的 `/v1/chat/config` 端点支持运行时配置热更新。

### 新增功能

- 新增 `/v1/chat/config` 端点，支持运行时更新 Anthropic 模型配置，无需重启服务。
- 新增 `/pta` 命令，提供 PyTorch-Ascend Agent，支持基于 `thread_id` 的多轮对话记忆，包含历史上下文拼装和逐轮持久化。
- 新增 `/experience` 命令，用于查看 Hermes 引擎内部状态（统计信息、记忆、技能、洞察、遗忘）。
- 新增 `/learn` 命令，支持用户将适配经验主动存入 Hermes 引擎。
- 新增 MCP Server 生命周期管理器（`app/tools/workflow/mcp_integration.py`），支持应用启动时自动拉起、退出时优雅关闭。
- 新增 Claude Code 会话历史监控（`app/services/claude_history_uploader.py`），通过后台 `HistoryWatcher` 监控并上传历史 Claude Code 会话。
- 新增 Claude Code 会话历史提炼（`app/services/claude_history_extractor.py`），从原始会话日志中提取结构化洞察。
- 新增基于 LangGraph `MemorySaver` 的多轮对话记忆（`app/core/memory.py`），支持对话状态的检查点保存与恢复。
- 新增技能注册中心（`app/tools/skills/registry.py`），用于跨框架管理、发现和版本化可复用技能。
- 新增 CI 流水线（`.github/workflows/ci.yml`），包含代码检查、单元测试和集成测试三个阶段。
- 新增 `requirements-dev.txt`，包含 `pytest`、`pytest-asyncio`、`pytest-cov` 和 `ruff`。
- 新增 Ascend NPU 环境的默认模型下载路径配置。
- 增强 LTS（日志/追踪/跨度）上报，包括 Claude Code 工具调用追踪和心跳监控。
- 新增 `trace_id` 全链路传播，实现请求生命周期的端到端可观测性。
- 新增运行时 LLM 密钥轮换支持——无需重启服务即可注入新的 Anthropic 访问令牌。

### 命令一览

| 命令 | 触发方式 | 说明 |
|---------|---------|-------------|
| `/claude` | 用户 / LLM / 自动 | 调用 Claude Code 技能 |
| `/verify` | 用户 / LLM / 自动 | 模型部署验证 |
| `/adapt` | 用户 / LLM / 自动 | 模型适配至 Ascend NPU |
| `/optimize` | 用户 / LLM / 自动 | vLLM-Ascend 性能优化 |
| `/quantify` | 用户 / LLM / 自动 | 模型量化 |
| `/commit` | 用户 / LLM / 自动 | 代码提交与推送 |
| `/search` | 用户 / LLM / 自动 | Ascend 模型搜索 |
| `/ai4s` | 用户 / LLM / 自动 | AI for Science 模型迁移 |
| `/deploy` | 用户 / LLM / 自动 | 模型部署 |
| `/doc` | 用户 / LLM / 自动 | 文档生成 |
| `/pta` | 用户 / LLM / 自动 | **PyTorch-Ascend Agent（新增）** |
| `/experience` | 仅用户 | **Hermes 引擎内省（新增）** |
| `/learn` | 仅用户 | **存储适配经验（新增）** |
| `/workflow` | 用户 / LLM / 自动 | 动态工作流编排 **（重写）** |

### 工作流引擎（`app/tools/workflow/`）

原有的静态关键词匹配意图解析器已替换为基于 DAG 的动态工作流引擎：

- **DynamicPlanner**（`planner.py`）：生成 LLM 驱动的多步骤执行计划，包含依赖解析、并行路径交叉验证和自适应目标分解。
- **WorkflowExecutor**（`executor.py`）：执行计划，支持并行步骤执行、可配置重试逻辑、通过 LangGraph `MemorySaver` 实现检查点/恢复，以及最大并发控制。
- **Brainstorming**（`brainstorming.py`）：多轮意图澄清，包含渐进式问题生成、模糊度评估、提案跟踪和设计文档输出。
- **MCP 集成**（`mcp_integration.py`）：桥接工作流引擎与外部 MCP 工具，允许工作流步骤调用 MCP Server 执行领域特定操作。
- **实时进度**：通过 `asyncio.Queue` 和 `progress_callback` 实现 SSE 进度推送，在工作流执行过程中提供每个步骤的实时状态更新。
- **执行报告**：端到端执行报告，包含逐步骤详情、耗时和执行结果。

### 经验引擎 — Hermes（`app/tools/experience/`）

一个自演进子系统，持久化并应用历史执行中积累的经验：

- **ExperienceStore**（`store.py`）：持久化层，基于 `memory.jsonl`、`skills.json` 和 `insights.jsonl`，支持原子写入和版本管理。
- **MemoryEngine**（`memory.py`）：语义记忆检索，根据当前上下文查找相关的历史经验。
- **SkillEngine**（`skill.py`）：从积累的经验中提取、版本化并应用技能。
- **NudgeEngine**（`nudge.py`）：基于历史经验的模式匹配，主动触发优化建议。
- **分层架构**（`layers/`）：可插拔的适配层，包括错误恢复、MCP 优化、模型适配和用户偏好。

### MCP Server（`app/tools/mcp_servers/`）

- **Cannbot Server**（`cannbot_server.py`）：通过 stdio 传输协议提供 CANNBot 技能查询，共 316 行服务端实现。
- **MS Agent Server**（`ms_agent_server.py` / `ms_agent_runner.py`）：Microsoft Agent MCP 集成，支持外部 Agent 工具调用。

### 性能与稳定性

- `app/main.py` 中的服务启动生命周期现在依次初始化：eval 日志补丁、MCP 管理器、Hermes 引擎和 Claude Code 历史监控器。
- 外部 Agent（search-agent、pta-agent）改为启动时克隆，不再本地持久化在仓库中，确保部署清洁。
- 新增 `_aiter_with_timeout` 辅助函数——`asyncio.wait_for` 无法包装异步生成器，因此实现了自定义超时包装器。
- 新增 `_CLAUDE_SKILL_TIMEOUT` 默认值，防止技能调用无限挂起。
- 优化 Claude Code 响应流式输出，实现合理的分块和进度上报。
- 新增全面的 `.gitignore` 规则，覆盖 `kernel_meta/`、`*.log`、生成的报告、`tmp/`、`PYEOF` 和 `fusion_result.json`。

### 代码清理

- 从版本控制中移除 96 个 `kernel_meta/` 文件（Ascend 编译缓存产物）。
- 移除 40+ 个 `__pycache__/` 目录及编译后的 `.pyc` 文件。
- 移除重复的技能目录和带 `-old` 后缀的技能变体（`optimizer-agent-old`、`verify-agent-old`）。
- 移除 `tmp/` 目录（20 个未被应用代码引用的测试/演示脚本）。
- 移除 `server.log`（版本控制中积压的 7MB+ 日志）。
- 移除 `reports/` 目录中的生成评估报告。
- 移除 `models/ai4s-basic` 技能（已合并至 skills 根目录下的 `ai4s-basic`）。
- 将 `optimizer-agent-old` 和 `verify-agent-old` 替换为 `optimizer-agent-plus` 和 `verify-agent-plus`。
- 移除 `test_page.html`（开发测试遗留产物）。

### Bug 修复

- 修复 `/workflow` 命令在 brainstorm 执行期间阻塞 SSE 流的问题——`_execute_workflow_with_context` 现在使用 `progress_callback` 配合 `asyncio.Queue` 实现实时步骤进度。
- 修复 brainstorm 循环中 LLM 可能陷入无限澄清循环的问题——新增最大轮次保护和收敛检测。
- 修复 search-agent 路径解析，使其在开发和生产环境中均能正常工作。
- 修复步骤超时处理——工作流步骤现在遵循每步超时配置。
- 修复 SSE 进度推送中进度队列在初始化前被访问导致的 `UnboundLocalError`。
- 修复 `sync_cannbot_skills.py` 语法错误（文档字符串中多余的引号）。
- 修复 9 个 SKILL.md 文件中硬编码的个人目录路径——替换为 `~`（Unix 家目录）和相对路径，同时保留标准的 Ascend NPU 工具链路径（`/home/Ascend/ascend-toolkit/`、`/usr/local/Ascend/`）。
- 修复 LangGraph 检查点中多轮对话 `thread_id` 的处理问题。
- 修复 `lazy_import` 缓存在热重载场景下可能持有过期模块引用的问题。
- 修复重复的 `_read_json` 函数——移除冗余副本。
- 修复 `assess_ambiguity` 为正确的异步实现，避免事件循环阻塞。
- 修复空提示词防护——框架现在拒绝空的用户提示词并返回明确错误，而非将其传递给 LLM。
- 修复 MCP Server 超时配置，防止外部工具调用无限挂起。
- 修复 `python3` 配置引用为平台无关写法。
- 修复 OpenAI 到 Anthropic 接口迁移中的多个边界问题——清理了过渡期间遗留的 OpenAI 格式请求处理。
- 修复工作流 Brainstorming 中对 ModelScope 链接的支持——来自 ModelScope 的 URL 现在能被正确识别和解析。
- 修复工作流引擎中对 `AskUserQuestion` 的处理——LLM 发出的交互式问题现在能正确传递至客户端。
- 修复技能门控逻辑——技能现在能根据用户请求上下文正确过滤。

### 依赖变更

| 依赖 | 原版本 | 当前版本 | 备注 |
|------------|-----------------|-----------------|-------|
| LLM SDK | langchain-openai | langchain-anthropic | Anthropic Claude 原生 |
| HTTP 客户端 | — | httpx>=0.28.0 | MCP 传输 |
| SSE 服务端 | — | sse-starlette>=2.0.0 | MCP 流式 |
| 开发工具 | — | pytest>=7.4.0, ruff | CI 流水线 |

其余依赖不变：FastAPI、Uvicorn、Pydantic、LangChain、LangGraph、python-dotenv、structlog、huaweicloudsdklts。

### 破坏性变更与迁移须知

- **环境变量重命名**：`OPENAI_API_KEY` → `ANTHROPIC_AUTH_TOKEN`，`OPENAI_BASE_URL` → `ANTHROPIC_BASE_URL`，`OPENAI_MODEL` → `ANTHROPIC_MODEL`。部署清单和 `.env` 文件必须更新。
- **API 路由重构**：`/api/system/health` 迁移至 `/_stcore/health`；`/api/chat/stream` 和 `/api/chat/brainstorm` 合并至 `/v1/chat/completions`。
- **配置端点需完整载荷**：`POST /v1/chat/config` 需要 `access_token`、`base_url` 和 `model` 三个字段。
- **`.env` 中新增可选配置项**：`HERMES_*`、`MCP_*`、`WORKFLOW_*`、`CLAUDE_HISTORY_*`——默认值开箱即用，生产环境可按需定制。
- **服务启动**现在要求 `run.py`（uvicorn 入口）位于项目根目录，运行于 18003 端口。
- **外部 Agent**（search-agent、pta-agent）不再内置于仓库中——改为启动时从各自源头克隆。

### 已知问题

- **eval/v0.1.0/ 目录**尚未填充——启动时的 eval 日志补丁会优雅跳过。
- **agents/ms_agent/** 仅为脚手架桩代码，尚无实际实现。
- **PTA Agent 多轮对话**在服务重启后可能丢失上下文，因为记忆为进程内存储，尚未持久化至磁盘。
- **Hermes 引擎用户偏好层**（`layers/user_prefs.py`）尚未集成到 Nudge 子系统中。
- **综合测试套件**当前通过率为 86.2%（160 个测试），剩余测试失败已记录并跟踪，计划在下一版本修复。

### 文档

- 更新 `README.md`，反映最新的 API 端点和环境变量参考。
- 更新 `.env.example`，包含 Anthropic Claude 配置、Hermes 引擎、MCP Server 和工作流相关选项。
