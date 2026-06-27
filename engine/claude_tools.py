from __future__ import annotations
"""claude-code / Claude Skill 调用工具模块。"""


import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from server.config import settings
from engine.memory import get_thread_messages

_SKILL_REGISTRY: dict[str, str] = {
    "adapt-agent": "AI-assisted model adaptation for Ascend NPU via vLLM-Ascend",
    "gitcode-publish": "GitCode 代码发布与提交助手",
    "npu-basic-migrate": "通用昇腾 NPU 模型迁移",
    "ascend-affinity-operator": "昇腾亲和算子优化",
    "ascend-history-to-skill": "从历史记录生成昇腾模型 skill",
    "ascend-optimization": "通用昇腾 PyTorch 优化",
    "optimizer-agent": "vLLM-Ascend 性能调优顾问",
    "quantify-agent": "昇腾推理工具链（量化、vLLM）",
    "verify-agent": "昇腾模型适配验证",
}

_SKILL_ALIASES: dict[str, str] = {
    "Ascend_Model_Verifier": "verify-agent",
    "Ascend_Model_Adapter": "adapt-agent",
    "Ascend_Model_Optimizer": "optimizer-agent",
    "Ascend_Model_Quantizer": "quantify-agent",
    "Ascend_Model_Deployer": "npu-basic-migrate",
    "Code_Reviewer": "Code_Reviewer",
    "PR_Analyzer": "PR_Analyzer",
}

_DEFAULT_ALLOWED_TOOLS = ["Bash", "Read", "Edit", "Write", "Glob", "Grep"]

_DEFAULT_DOWNLOAD_PATH_HINT = "\n\n注意：如果任务中需要下载文件，请默认将文件保存到 /data 目录下。"

# 匹配包含 UserWarning、warnings.warn 的行并删除
_FILTER_LINE_RE = re.compile(
    r"^.*(?:UserWarning|warnings\.warn).*\n?",
    re.MULTILINE | re.IGNORECASE,
)

# Claude Skill 单次调用默认超时（秒）
_CLAUDE_SKILL_TIMEOUT: int = 300


def _find_claude_cli() -> str | None:
    """动态查找 claude CLI 路径。"""
    cli_path = shutil.which("claude")
    if cli_path:
        return cli_path
    # 常见 npm global 安装路径
    for candidate in [
        Path.home() / "npm-global" / "bin" / "claude",
        Path("/usr/local/bin/claude"),
        Path("/usr/bin/claude"),
    ]:
        if candidate.exists():
            return str(candidate)
    return None


async def _aiter_with_timeout(aiter, timeout: float):
    """为异步生成器添加超时机制。

    asyncio.wait_for 不能直接包装 async generator，
    需要逐元素用 wait_for 包装 __anext__() 调用。
    """
    it = aiter.__aiter__()
    while True:
        try:
            yield await asyncio.wait_for(it.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return


def _filter_warnings(text: str) -> str:
    """过滤包含 warning 或常见错误信息的行。"""
    if not text or not isinstance(text, str):
        return text
    filtered = _FILTER_LINE_RE.sub("", text)
    # 压缩因删除行而产生的多余空行
    filtered = re.sub(r"\n{3,}", "\n\n", filtered)
    return filtered.strip("\n")


def _chunk_text(text: str, max_chunk_size: int = 80) -> list[str]:
    """将长文本拆分成适当大小的 chunk，优先按行拆分，超长行按 max_chunk_size 拆分。"""
    if not text:
        return []
    chunks: list[str] = []
    for line in text.splitlines(keepends=True):
        if len(line) <= max_chunk_size:
            chunks.append(line)
        else:
            for i in range(0, len(line), max_chunk_size):
                chunks.append(line[i : i + max_chunk_size])
    return chunks


_SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


def _find_skill_md(skill_name: str) -> Path | None:
    """Find the SKILL.md file for a given skill name.

    Searches by directory name pattern: skills/*/skill_name/SKILL.md
    Falls back to scanning all SKILL.md frontmatter for matching name.
    """
    if not _SKILLS_DIR.is_dir():
        return None
    # Fast path: search by directory name
    candidates = list(_SKILLS_DIR.rglob(f"*/{skill_name}/SKILL.md"))
    if candidates:
        return candidates[0]
    # Slow path: scan frontmatter (first-run only, cached in registry)
    for skill_md in _SKILLS_DIR.rglob("SKILL.md"):
        try:
            text = skill_md.read_text(encoding="utf-8", errors="ignore")
            if text.startswith("---"):
                end = text.find("---", 3)
                if end != -1:
                    front = text[3:end]
                    for line in front.splitlines():
                        if line.strip().startswith("name:") and skill_name in line:
                            return skill_md
        except Exception:
            continue
    return None


def _load_skill_content(skill_name: str) -> str | None:
    """Load SKILL.md content for a skill, stripping YAML frontmatter."""
    skill_md = _find_skill_md(skill_name)
    if not skill_md:
        return None
    try:
        text = skill_md.read_text(encoding="utf-8", errors="ignore")
        # Strip YAML frontmatter
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                text = text[end + 3:].strip()
        return text
    except Exception:
        return None


def _resolve_skill_name(skill_name: str) -> str | None:
    if skill_name in _SKILL_REGISTRY:
        return skill_name
    if skill_name in _SKILL_ALIASES:
        return _SKILL_ALIASES[skill_name]
    return None


def _build_full_prompt(skill_name: str, prompt: str) -> str:
    """Build a prompt that includes the skill definition for the model to follow."""
    skill_content = _load_skill_content(skill_name)
    if skill_content:
        return (
            f"你是一个名为 {skill_name} 的 AI 助手。请严格按照以下指令完成任务：\n\n"
            f"## {skill_name} Skill 指令\n\n"
            f"{skill_content}\n\n"
            f"---\n\n"
            f"## 用户任务\n\n"
            f"{prompt}{_DEFAULT_DOWNLOAD_PATH_HINT}"
        )
    else:
        return (
            f"请作为 {skill_name} 助手来完成以下任务。"
            f"你需要运用你的知识和工具来完成任务：\n\n"
            f"{prompt}{_DEFAULT_DOWNLOAD_PATH_HINT}"
        )


def _format_history_as_context(messages: list[Any], max_turns: int = 10, max_chars: int = 4000) -> str:
    """将 LangChain 消息列表格式化为 Claude 可读的上下文文本。"""
    if not messages:
        return ""

    # 估算每轮 2 条消息，取最近的若干条
    recent = messages[-max_turns * 2 :]

    lines: list[str] = []
    for msg in recent:
        role = type(msg).__name__
        if role == "SystemMessage":
            role_label = "系统"
        elif role == "HumanMessage":
            role_label = "用户"
        elif role == "AIMessage":
            role_label = "助手"
            # 跳过纯工具调用消息（无文本内容）
            content = getattr(msg, "content", "")
            if not content and getattr(msg, "tool_calls", None):
                continue
        elif role == "ToolMessage":
            name = getattr(msg, "name", "unknown")
            role_label = f"工具({name})"
        else:
            continue

        content = str(getattr(msg, "content", "") or "")
        if not content:
            continue
        # 截断过长内容，避免上下文爆炸
        if len(content) > 800:
            content = content[:800] + "...[截断]"
        lines.append(f"【{role_label}】{content}")

    context = "\n".join(lines)
    if len(context) > max_chars:
        context = context[-max_chars:]
        first_nl = context.find("\n")
        if first_nl != -1:
            context = context[first_nl + 1 :]
        context = "...[前文省略]\n" + context

    if context:
        return f"以下是当前会话的上下文供你参考：\n\n{context}\n\n---\n\n"
    return ""


def _collect_claude_output(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        msg_type = type(message).__name__
        if msg_type == "ResultMessage":
            result_text = getattr(message, "result", "")
            result_text = _filter_warnings(result_text)
            if result_text:
                parts.append(f"**最终结果**：\n{result_text}")
        elif msg_type == "AssistantMessage":
            content = getattr(message, "content", [])
            if content and isinstance(content, list):
                for block in content:
                    block_type = type(block).__name__
                    if block_type == "ThinkingBlock":
                        thinking = getattr(block, "thinking", "")
                        thinking = _filter_warnings(thinking)
                        if thinking:
                            parts.append(f"> **思考**：{thinking}")
                    elif block_type == "ToolUseBlock":
                        tool_name = getattr(block, "name", "Unknown")
                        tool_input = getattr(block, "input", {})
                        if tool_name == "AskUserQuestion":
                            if isinstance(tool_input, dict):
                                questions = tool_input.get("questions", tool_input.get("question", []))
                                if isinstance(questions, list) and questions:
                                    q_parts = ["**需要补充以下信息**："]
                                    for i, q in enumerate(questions):
                                        if isinstance(q, dict):
                                            q_text = q.get("question", str(q))
                                            q_header = q.get("header", f"问题 {i+1}")
                                            q_parts.append(f"{i+1}. **{q_header}**: {q_text}")
                                        else:
                                            q_parts.append(f"{i+1}. {q}")
                                    parts.append("\n".join(q_parts))
                                elif isinstance(questions, str):
                                    parts.append(f"**需要补充信息**: {questions}")
                            continue
                        tool_info = f"**使用工具**：`{tool_name}`"
                        cmd = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
                        if cmd:
                            tool_info += f"\n```bash\n{cmd}\n```"
                        parts.append(tool_info)
                    elif block_type == "TextBlock":
                        text = getattr(block, "text", "")
                        text = _filter_warnings(text)
                        if text:
                            parts.append(text)
        elif msg_type == "UserMessage":
            tool_result = getattr(message, "tool_use_result", None)
            if tool_result and isinstance(tool_result, dict):
                stdout = tool_result.get("stdout", "")
                stderr = tool_result.get("stderr", "")
                stdout = _filter_warnings(stdout)
                stderr = _filter_warnings(stderr)
                if stdout or stderr:
                    output_text = ""
                    if stdout:
                        output_text += f"**标准输出**：\n```\n{stdout}\n```"
                    if stderr:
                        output_text += f"\n**错误输出**：\n```\n{stderr}\n```"
                    parts.append(output_text)
            elif tool_result and isinstance(tool_result, str):
                tool_result = _filter_warnings(tool_result)
                if tool_result:
                    parts.append(f"```\n{tool_result}\n```")
    return "\n\n".join(parts) if parts else "Claude Code 执行完成，未返回内容。"


def _load_claude_settings() -> dict:
    settings_path = Path.home() / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            return json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


async def _invoke_claude_skill_async(skill_name: str, prompt: str) -> str:
    if not skill_name and not prompt:
        return "[claude-code] 错误：未提供 skill_name 或 prompt，无法执行。"
    if skill_name:
        full_prompt = _build_full_prompt(skill_name, prompt)
    else:
        full_prompt = prompt

    try:
        # The project's mcp/ directory shadows the pip-installed mcp package
        import sys as _sys
        _cwd = os.getcwd()
        _cwd_in_path = _cwd in _sys.path
        if _cwd_in_path:
            _sys.path.remove(_cwd)

        from claude_agent_sdk import ClaudeAgentOptions, query as claude_query

        if _cwd_in_path:
            _sys.path.insert(0, _cwd)

        creds = _resolve_sdk_credentials()

        options_kwargs: dict[str, Any] = {
            "allowed_tools": _DEFAULT_ALLOWED_TOOLS,
            "permission_mode": settings.claude_permission_mode,
        }
        if creds.get("ANTHROPIC_MODEL"):
            options_kwargs["model"] = creds["ANTHROPIC_MODEL"]
        options_kwargs["env"] = creds

        options = ClaudeAgentOptions(**options_kwargs)
        messages: list[Any] = []
        try:
            async for message in _aiter_with_timeout(
                claude_query(prompt=full_prompt, options=options),
                _CLAUDE_SKILL_TIMEOUT,
            ):
                messages.append(message)
        except asyncio.TimeoutError:
            return f"[claude-code] 调用超时（>{_CLAUDE_SKILL_TIMEOUT}s）"

        return _collect_claude_output(messages)
    except ModuleNotFoundError:
        pass  # 降级到 CLI

    # ── 降级：使用 claude CLI（npm） ─────────────────────────────────
    claude_cli = _find_claude_cli()
    if claude_cli:
        # 子进程 env 从当前环境继承，确保 PATH 等正确
        subprocess_env = os.environ.copy()
        subprocess_env.update(creds)
        subprocess_env.update({
            "CLAUDE_CODE_SIMPLE": "1",
            "HOME": str(Path.home()),
        })

        proc = await asyncio.create_subprocess_exec(
            claude_cli,
            "--print",
            "--no-session-persistence",
            "--bare",
            "--output-format", "text",
            full_prompt,
            env=subprocess_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=_CLAUDE_SKILL_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            return f"[claude-code] CLI 调用超时（>{_CLAUDE_SKILL_TIMEOUT}s）"

        if proc.returncode == 0:
            result = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            result = _filter_warnings(result)
            return result.strip()
        else:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            if stderr_text:
                return f"[claude-code] CLI 退出码 {proc.returncode}：{stderr_text[:500]}"

    return "[claude-code] 环境未安装 claude-agent-sdk 且找不到 claude CLI，请执行 `pip install claude-agent-sdk`。"


@tool
def list_claude_skills() -> str:
    """列出当前环境可用的 Claude Skill 名称列表及说明。"""
    lines = ["当前可用的 Claude Skill 列表："]
    for name, desc in _SKILL_REGISTRY.items():
        lines.append(f"- **{name}**：{desc}")
    lines.append("\n使用 `invoke_claude_skill` 调用指定 Skill，传入 `skill_name` 和具体 `prompt`。*")
    return "\n".join(lines)


@tool
async def invoke_claude_skill(skill_name: str, prompt: str) -> str:
    """调用指定 Claude Skill，传入自然语言 prompt，由 Claude Code Agent 实际执行工具链完成任务。

    Args:
        skill_name: Skill 名称，例如 verify-agent、optimizer-agent 等。
        prompt: 传给 Skill 的提示内容。
    """
    canonical = _resolve_skill_name(skill_name)
    if canonical is None:
        # 未知的 skill，将整个输入透传给 Claude SDK 处理
        full_prompt = skill_name
        if prompt:
            full_prompt += " " + prompt
        try:
            return await _invoke_claude_skill_async("", full_prompt)
        except Exception as e:
            return f"[claude-code] 调用失败：{type(e).__name__}: {e}"
    try:
        return await _invoke_claude_skill_async(canonical, prompt)
    except Exception as e:
        return f"[claude-code] 调用失败：{type(e).__name__}: {e}"


def _resolve_sdk_credentials() -> dict[str, str]:
    """Resolve API credentials with same priority as _resolve_anthropic_config().

    Priority: server settings > ~/.claude/settings.json > system env vars.
    Returns dict with env vars the Claude CLI actually reads (ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL, ANTHROPIC_MODEL).
    """
    claude_settings = _load_claude_settings()
    claude_env = claude_settings.get("env", {})

    api_key = (
        settings.anthropic_auth_token
        or claude_env.get("ANTHROPIC_AUTH_TOKEN")
        or os.getenv("ANTHROPIC_AUTH_TOKEN")
        or ""
    )
    base_url = (
        settings.anthropic_base_url
        or claude_env.get("ANTHROPIC_BASE_URL")
        or os.getenv("ANTHROPIC_BASE_URL")
        or ""
    )
    model = (
        settings.anthropic_model
        or claude_env.get("ANTHROPIC_MODEL")
        or os.getenv("ANTHROPIC_MODEL")
        or None
    )
    result: dict[str, str] = {}
    # Claude CLI reads ANTHROPIC_API_KEY from environment for authentication
    if api_key:
        result["ANTHROPIC_API_KEY"] = api_key
    if base_url:
        result["ANTHROPIC_BASE_URL"] = base_url
    if model:
        result["ANTHROPIC_MODEL"] = model
    return result


async def _stream_claude_skill_async(skill_name: str, prompt: str):
    """Stream Claude Skill output, yielding text chunks."""
    if skill_name:
        full_prompt = _build_full_prompt(skill_name, prompt)
    else:
        full_prompt = prompt

    creds = _resolve_sdk_credentials()

    # Use Python SDK
    try:
        # The project's mcp/ directory shadows the pip-installed mcp package
        # needed by claude_agent_sdk. Temporarily remove cwd from sys.path.
        import sys as _sys
        _cwd = os.getcwd()
        _cwd_in_path = _cwd in _sys.path
        if _cwd_in_path:
            _sys.path.remove(_cwd)

        from claude_agent_sdk import ClaudeAgentOptions, query as claude_query

        # Restore sys.path
        if _cwd_in_path:
            _sys.path.insert(0, _cwd)

        options_kwargs: dict[str, Any] = {
            "allowed_tools": _DEFAULT_ALLOWED_TOOLS,
            "permission_mode": settings.claude_permission_mode,
        }
        if creds.get("ANTHROPIC_MODEL"):
            options_kwargs["model"] = creds["ANTHROPIC_MODEL"]
        # Pass credentials via env to override ~/.claude/settings.json defaults
        options_kwargs["env"] = creds

        options = ClaudeAgentOptions(**options_kwargs)

        async for message in claude_query(prompt=full_prompt, options=options):
            msg_type = type(message).__name__

            if msg_type == "SystemMessage":
                continue

            if msg_type == "ResultMessage":
                result_text = getattr(message, "result", "")
                result_text = _filter_warnings(result_text)
                if result_text:
                    for chunk in _chunk_text(result_text):
                        yield chunk

            elif msg_type == "AssistantMessage":
                content = getattr(message, "content", [])
                if content and isinstance(content, list):
                    for block in content:
                        block_type = type(block).__name__
                        if block_type == "ThinkingBlock":
                            thinking = getattr(block, "thinking", "")
                            thinking = _filter_warnings(thinking)
                            if thinking:
                                yield "\n> **Thinking:** "
                                yield thinking
                        elif block_type == "ToolUseBlock":
                            tool_name = getattr(block, "name", "Unknown")
                            tool_input = getattr(block, "input", {})
                            if tool_name == "AskUserQuestion":
                                yield f"\n> **Q:** {tool_input}"
                                continue
                            tool_info = f"\n> **Tool:** `{tool_name}`\n"
                            cmd = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
                            if cmd:
                                tool_info += f"> ```\n> {cmd}\n> ```\n"
                            yield tool_info
                        elif block_type == "TextBlock":
                            text = getattr(block, "text", "")
                            text = _filter_warnings(text)
                            if text:
                                for chunk in _chunk_text(text):
                                    yield chunk

            elif msg_type == "UserMessage":
                tool_result = getattr(message, "tool_use_result", None)
                if tool_result and isinstance(tool_result, dict):
                    stdout = tool_result.get("stdout", "")
                    stderr = tool_result.get("stderr", "")
                    if stdout:
                        yield f"\n```\n{_filter_warnings(stdout)}\n```\n"
                    if stderr:
                        yield f"\n```\n{_filter_warnings(stderr)}\n```\n"
                elif tool_result and isinstance(tool_result, str):
                    yield f"\n```\n{_filter_warnings(tool_result)}\n```\n"

            elif msg_type == "StreamEvent":
                delta = getattr(message, "delta", None)
                if delta and isinstance(delta, dict):
                    text = delta.get("text", "")
                    if text:
                        yield str(text)

        return
    except ModuleNotFoundError as e:
        yield f"\n[DEBUG] ModuleNotFoundError: {e}\n"
        pass
    except Exception as e:
        yield f"\n[SDK error: {type(e).__name__}: {e}]\n"
        return

    # Fallback: use claude CLI
    claude_cli = _find_claude_cli()
    if claude_cli:
        subprocess_env = os.environ.copy()
        subprocess_env.update(creds)
        subprocess_env.update({
            "CLAUDE_CODE_SIMPLE": "1",
            "HOME": str(Path.home()),
        })

        proc = await asyncio.create_subprocess_exec(
            claude_cli,
            "--print",
            "--no-session-persistence",
            "--bare",
            "--output-format", "text",
            full_prompt,
            env=subprocess_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert proc.stdout is not None

        try:
            while True:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=_CLAUDE_SKILL_TIMEOUT)
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                if text:
                    for chunk in _chunk_text(text):
                        yield chunk
                        await asyncio.sleep(0)
        except asyncio.TimeoutError:
            proc.kill()
            yield f"\n[claude-code] CLI timeout (>{_CLAUDE_SKILL_TIMEOUT}s)"
            return

        await proc.wait()
        if proc.returncode != 0:
            stderr_output = (await proc.stderr.read()).decode("utf-8", errors="replace") if proc.stderr else ""
            if stderr_output:
                yield f"\n[claude-code] CLI exit {proc.returncode}: {stderr_output[:500]}"
            return
        return

    yield "[claude-code] No SDK and no CLI available."


async def stream_claude_skill(skill_name: str, prompt: str):
    """流式调用 Claude Skill 的公共入口。

    根据 skill_name 解析后，逐条 yield 处理后的文本片段。
    """
    canonical = _resolve_skill_name(skill_name)
    if canonical is None:
        full_prompt = skill_name
        if prompt:
            full_prompt += " " + prompt
        async for chunk in _stream_claude_skill_async("", full_prompt):
            yield chunk
        return

    async for chunk in _stream_claude_skill_async(canonical, prompt):
        yield chunk


def get_tools() -> list:
    return [list_claude_skills, invoke_claude_skill]
