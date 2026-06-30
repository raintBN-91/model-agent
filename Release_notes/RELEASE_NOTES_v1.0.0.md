# Release Notes

## 2026.06.30

This release represents a major architectural upgrade that transitions the backend from a simple OpenAI-compatible chat proxy into a full-featured model optimization framework with dynamic workflow orchestration, a self-evolving experience engine, and MCP (Model Context Protocol) integration.

### Highlights

- **Dynamic Workflow Engine**: Replaced the static intent-resolver with a DAG-based `DynamicPlanner` + `WorkflowExecutor` supporting LLM-generated plans, parallel step execution, retry logic, checkpointing, and real-time SSE progress streaming.
- **Hermes Self-Evolving Engine**: New experience subsystem (`app/tools/experience/`) that persists adaptation memories, skills, and insights — enabling the framework to learn from past executions and proactively suggest optimizations.
- **MCP Integration**: Model Context Protocol support with Cannbot and MS Agent MCP servers, lifecycle management, and workflow engine integration for tool calling.
- **PTA Agent**: Dedicated PyTorch-Ascend agent (`/pta`) with thread-aware multi-turn conversation memory for `torch_npu` development tasks.
- **Anthropic Claude Native**: Switched the LLM provider from MiniMax/MoonShot (OpenAI-compatible) to Anthropic Claude, with runtime configuration hot-reload via the new `/v1/chat/config` endpoint.

### Features

- Added `/v1/chat/config` endpoint for runtime Anthropic model configuration updates without service restart.
- Added `/pta` command for PyTorch-Ascend agent with `thread_id`-based multi-turn conversation memory, including history context assembly and per-exchange persistence.
- Added `/experience` command for introspecting Hermes engine state (stats, memory, skills, insights, forget).
- Added `/learn` command for user-teaching — storing adaptation experiences into the Hermes engine.
- Added MCP server lifecycle manager (`app/tools/workflow/mcp_integration.py`) with auto-startup on application launch and graceful shutdown.
- Added Claude Code session history monitoring (`app/services/claude_history_uploader.py`) with a background `HistoryWatcher` that monitors and uploads historical Claude Code sessions.
- Added Claude Code session history extraction (`app/services/claude_history_extractor.py`) for distilling structured insights from raw session logs.
- Added multi-turn conversation memory via LangGraph `MemorySaver` (`app/core/memory.py`) for checkpointing and restoring conversation state.
- Added skill registry (`app/tools/skills/registry.py`) for managing, discovering, and versioning reusable skills across the framework.
- Added CI pipeline (`.github/workflows/ci.yml`) with lint, unit test, and integration test stages.
- Added `requirements-dev.txt` with `pytest`, `pytest-asyncio`, `pytest-cov`, and `ruff`.
- Added default model download path configuration for Ascend NPU environments.
- Added LTS (Log/Trace/Span) reporting enhancements including Claude Code tool call tracing and heartbeat monitoring.
- Added `trace_id` propagation throughout the request lifecycle for end-to-end observability.
- Added runtime LLM key rotation support — new Anthropic access tokens can be injected without restarting the service.

### Commands

| Command | Trigger | Description |
|---------|---------|-------------|
| `/claude` | User / LLM / Auto | Invoke Claude Code skills |
| `/verify` | User / LLM / Auto | Model deployment verification |
| `/adapt` | User / LLM / Auto | Model adaptation to Ascend NPU |
| `/optimize` | User / LLM / Auto | vLLM-Ascend performance optimization |
| `/quantify` | User / LLM / Auto | Model quantization |
| `/commit` | User / LLM / Auto | Code commit and push |
| `/search` | User / LLM / Auto | Ascend model search |
| `/ai4s` | User / LLM / Auto | AI for Science model migration |
| `/deploy` | User / LLM / Auto | Model deployment |
| `/doc` | User / LLM / Auto | Documentation generation |
| `/pta` | User / LLM / Auto | **PyTorch-Ascend agent (new)** |
| `/experience` | User only | **Hermes engine introspection (new)** |
| `/learn` | User only | **Store adaptation experience (new)** |
| `/workflow` | User / LLM / Auto | Dynamic workflow orchestration **(rewritten)** |

### Workflow Engine (`app/tools/workflow/`)

The static keyword-matching intent resolver has been replaced with a DAG-based dynamic workflow engine:

- **DynamicPlanner** (`planner.py`): Generates LLM-driven multi-step plans with dependency resolution, cross-validation between parallel paths, and adaptive goal decomposition.
- **WorkflowExecutor** (`executor.py`): Executes plans with parallel step execution, configurable retry logic, checkpoint/restore via LangGraph `MemorySaver`, and max concurrency control.
- **Brainstorming** (`brainstorming.py`): Multi-round intent clarification with progressive question generation, ambiguity assessment, proposal tracking, and design-doc output.
- **MCP Integration** (`mcp_integration.py`): Bridges the workflow engine with external MCP tools, allowing workflow steps to call MCP servers for domain-specific operations.
- **Real-time Progress**: SSE progress streaming via `asyncio.Queue` and `progress_callback`, providing per-step status updates during workflow execution.
- **Execution Report**: End-to-end execution report with step-by-step details, timing, and outcomes.

### Experience Engine — Hermes (`app/tools/experience/`)

A self-evolving subsystem that persists and applies learnings from past executions:

- **ExperienceStore** (`store.py`): Persistence layer backed by `memory.jsonl`, `skills.json`, and `insights.jsonl` with atomic writes and versioning.
- **MemoryEngine** (`memory.py`): Semantic memory retrieval for finding relevant past experiences based on current context.
- **SkillEngine** (`skill.py`): Skill extraction, versioning, and application from accumulated experiences.
- **NudgeEngine** (`nudge.py`): Proactive optimization suggestions triggered by pattern matching against past experiences.
- **Layer Architecture** (`layers/`): Pluggable adaptation layers including error recovery, MCP optimization, model adaptation, and user preferences.

### MCP Servers (`app/tools/mcp_servers/`)

- **Cannbot Server** (`cannbot_server.py`): CANNBot skill query via stdio transport with 316 lines of server implementation.
- **MS Agent Server** (`ms_agent_server.py` / `ms_agent_runner.py`): Microsoft Agent MCP integration for external agent tool calling.

### Performance & Stability

- Server startup lifecycle in `app/main.py` now initializes: eval logging patching, MCP manager, Hermes engine, and Claude Code history watcher during the application lifespan.
- External agents (search-agent, pta-agent) are now cloned at startup rather than persisted locally in the repository, ensuring clean deployments.
- Added `_aiter_with_timeout` helper — `asyncio.wait_for` cannot wrap async generators, so a custom timeout wrapper was implemented for stream generators.
- Added `_CLAUDE_SKILL_TIMEOUT` default value to prevent hanging skill invocations.
- Optimized stream output for Claude Code responses with proper chunking and progress reporting.
- Added comprehensive `.gitignore` patterns covering `kernel_meta/`, `*.log`, generated reports, `tmp/`, `PYEOF`, and `fusion_result.json`.

### Code Cleanup

- Removed 96 `kernel_meta/` files (Ascend compilation cache artifacts) from version control.
- Removed 40+ `__pycache__/` directories and compiled `.pyc` files.
- Removed duplicate skills directories and `-old` suffixed skill variants (`optimizer-agent-old`, `verify-agent-old`).
- Removed `tmp/` directory (20 test/demo scripts not referenced by application code).
- Removed `server.log` (7MB+ logs from version control).
- Removed `reports/` directory with generated evaluation reports.
- Removed `models/ai4s-basic` skill (consolidated into `ai4s-basic` under skills root).
- Replaced `optimizer-agent-old` and `verify-agent-old` with `optimizer-agent-plus` and `verify-agent-plus`.
- Removed `test_page.html` (development testing artifact).

### Bug Fixes

- Fixed `/workflow` command blocking SSE stream during brainstorm execution — `_execute_workflow_with_context` now uses `progress_callback` with `asyncio.Queue` for real-time step progress.
- Fixed brainstorm loop where the LLM could enter an infinite clarification cycle — added max-round guard and convergence detection.
- Fixed search-agent path resolution to work correctly in both development and production environments.
- Fixed step timeout handling — workflow steps now respect per-step timeout configuration.
- Fixed `UnboundLocalError` in SSE progress streaming when the progress queue was accessed before initialization.
- Fixed `sync_cannbot_skills.py` syntax error (extra quote in docstring).
- Fixed hardcoded personal home directory paths across 9 SKILL.md files — replaced with `~` (Unix home) and relative paths, while preserving standard Ascend NPU toolkit paths (`/home/Ascend/ascend-toolkit/`, `/usr/local/Ascend/`).
- Fixed multi-turn conversation `thread_id` handling in LangGraph checkpointer.
- Fixed `lazy_import` caching to avoid stale module references across hot-reload scenarios.
- Fixed duplicate `_read_json` function — removed the redundant copy.
- Fixed `assess_ambiguity` to be properly async, preventing event loop blocking.
- Fixed empty prompt guard — the framework now rejects empty user prompts with a clear error instead of passing them to the LLM.
- Fixed MCP server timeout configuration to prevent indefinite hangs on external tool calls.
- Fixed `python3` config reference to be platform-independent.
- Fixed OpenAI-to-Anthropic interface migration bugs — several edge cases where the transition left stale OpenAI-format request handling.
- Fixed ModelScope link support in workflow brainstorming — URLs from ModelScope are now correctly recognized and parsed.
- Fixed `AskUserQuestion` handling in the workflow engine — interactive questions from the LLM are now properly relayed to the client.
- Fixed skill gating logic — skills are now correctly filtered based on the user's request context.

### Dependencies

| Dependency | Previous Version | Current Version | Notes |
|------------|-----------------|-----------------|-------|
| LLM SDK | langchain-openai | langchain-anthropic | Anthropic Claude native |
| HTTP Client | — | httpx>=0.28.0 | MCP transport |
| SSE Server | — | sse-starlette>=2.0.0 | MCP streaming |
| Dev Tools | — | pytest>=7.4.0, ruff | CI pipeline |

Other dependencies unchanged: FastAPI, Uvicorn, Pydantic, LangChain, LangGraph, python-dotenv, structlog, huaweicloudsdklts.

### Breaking Changes & Migration Notes

- **Environment variables renamed**: `OPENAI_API_KEY` → `ANTHROPIC_AUTH_TOKEN`, `OPENAI_BASE_URL` → `ANTHROPIC_BASE_URL`, `OPENAI_MODEL` → `ANTHROPIC_MODEL`. Deployment manifests and `.env` files must be updated.
- **API routes restructured**: `/api/system/health` migrated to `/_stcore/health`; `/api/chat/stream` and `/api/chat/brainstorm` merged into `/v1/chat/completions`.
- **Config endpoint requires full payload**: `POST /v1/chat/config` requires `access_token`, `base_url`, and `model` fields.
- **New optional config sections in `.env`**: `HERMES_*`, `MCP_*`, `WORKFLOW_*`, `CLAUDE_HISTORY_*` — sensible defaults work out of the box, but can be customized for production.
- **Server startup** now expects `run.py` (uvicorn entry point) at project root, running on port 18003.
- **External agents** (search-agent, pta-agent) are no longer bundled in the repository — they are cloned at startup from their respective sources.

### Known Issues

- **eval/v0.1.0/ directory** is not yet populated — eval logging patching at startup is gracefully skipped.
- **agents/ms_agent/** is a scaffolding stub with no active implementation.
- **PTA Agent multi-turn conversations** may lose context if the server restarts, as memory is in-process and not yet persisted to disk.
- **Hermes engine user preference layer** (`layers/user_prefs.py`) is not yet integrated into the nudge subsystem.
- **Comprehensive test suite** achieves 86.2% pass rate (160 tests); remaining test failures are documented and tracked for the next release.

### Documentation

- Updated `README.md` with current API endpoints and environment variable reference.
- Updated `.env.example` with Anthropic Claude configuration, Hermes engine, MCP server, and workflow options.

