"""msAgent MCP Server — 启动时拉取仓库，将 Agent 能力注册为 MCP 工具。

仓库: https://gitcode.com/hewenbo/msagent

工作方式:
  1. 启动时 git clone --depth=1 到临时目录
  2. 扫描 SKILL.md / agent 配置文件
  3. 每个能力注册为一个 MCP tool
  4. 关闭时清理临时目录

MCP 协议（stdio transport）:
    stdin/stdout 通过 JSON-RPC 2.0 通信
    - tools/list → 列出所有可用的 msAgent 工具
    - tools/call → 执行指定的工具
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

DEFAULT_REPO_URL = "https://gitcode.com/hewenbo/msagent.git"
DEFAULT_BRANCH = "main"


def _clone_repo(url: str, dest: Path, branch: str = DEFAULT_BRANCH) -> None:
    """git clone --depth=1 到目标目录。"""
    result = subprocess.run(
        ["git", "clone", "--depth=1", "-b", branch, url, str(dest)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        # 尝试不加分支参数（默认分支可能不叫 main）
        result = subprocess.run(
            ["git", "clone", "--depth=1", url, str(dest)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"克隆 msAgent 仓库失败: {result.stderr.strip() or result.stdout.strip()}"
            )


def _discover_skills(repo_dir: Path) -> list[dict[str, str]]:
    """在克隆的仓库中查找 SKILL.md/AGENT.md 文件。"""
    seen: set[str] = set()
    skills: list[dict[str, str]] = []

    # 1. 查找所有 SKILL.md
    for skill_md in sorted(repo_dir.rglob("SKILL.md")):
        parsed = _parse_frontmatter(skill_md)
        if parsed and parsed["name"] not in seen:
            seen.add(parsed["name"])
            skills.append(parsed)

    # 2. 查找所有 AGENT.md / AGENT.yaml
    if not skills:
        for agent_file in sorted(repo_dir.rglob("AGENT*")):
            if agent_file.suffix in (".md", ".yaml", ".yml"):
                parsed = _parse_agent_config(agent_file)
                if parsed and parsed["name"] not in seen:
                    seen.add(parsed["name"])
                    skills.append(parsed)

    # 3. 如果仓库没有标准配置文件，暴露为一个通用 agent 工具
    if not skills:
        skills.append({
            "name": "ms_agent_ask",
            "description": "向 msAgent 发送自然语言查询，由 msAgent 自主规划并执行任务",
            "skill_dir": str(repo_dir),
        })

    skills.sort(key=lambda s: s["name"])
    return skills


def _parse_frontmatter(md_path: Path) -> dict[str, str] | None:
    """解析 Markdown 的 YAML frontmatter，提取 name 和 description。"""
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    if not text.startswith("---"):
        return None

    end = text.find("---", 3)
    if end == -1:
        return None

    front = text[3:end]
    name: str | None = None
    description = ""
    in_desc = False

    for line in front.splitlines():
        stripped = line.strip()
        if stripped.startswith("name:"):
            name = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            in_desc = False
        elif stripped.startswith("description:"):
            raw = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            description = raw
            in_desc = True
        elif in_desc and (stripped.startswith("-") or stripped.startswith(">")
                          or stripped.startswith("https://") or stripped.startswith("http://")):
            description += " " + stripped.lstrip("- >").strip()
        elif in_desc and stripped:
            in_desc = False

    if name:
        return {"name": name, "description": description or name}
    return None


def _parse_agent_config(config_path: Path) -> dict[str, str] | None:
    """解析 AGENT 配置文件，提取 agent 名称和描述。"""
    try:
        text = config_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    name = config_path.stem  # AGENT.md → "AGENT"
    description = text.split("\n")[0].strip().strip("#").strip()
    if not description:
        description = name

    return {"name": name, "description": description, "skill_dir": str(config_path.parent)}


def _get_repo_info(repo_dir: Path) -> dict[str, Any]:
    """获取仓库基本信息。"""
    info: dict[str, Any] = {
        "repo_size": 0,
        "file_count": 0,
        "top_level": [],
    }
    try:
        for f in repo_dir.iterdir():
            if not f.name.startswith("."):
                info["top_level"].append(f.name)
                if f.is_file():
                    info["repo_size"] += f.stat().st_size
                    info["file_count"] += 1
                elif f.is_dir():
                    info["file_count"] += sum(1 for _ in f.rglob("*") if _.is_file())
    except Exception:
        pass
    return info


async def _run_skill(skill_name: str, prompt: str) -> str:
    """通过 Claude Code 调用 skill。"""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from engine.claude_tools import stream_claude_skill

        chunks: list[str] = []
        async for chunk in stream_claude_skill(skill_name, prompt):
            if chunk:
                chunks.append(chunk)
        return "".join(chunks) if chunks else "(无输出)"
    except ImportError as e:
        return f"[msAgent] 无法加载 claude-agent-sdk: {e}"
    except Exception as e:
        return f"[msAgent] 调用失败: {type(e).__name__}: {e}\n{traceback.format_exc()}"


async def _run_repo_agent(repo_dir: Path, query: str) -> str:
    """对克隆的 msAgent 仓库执行通用查询。

    将仓库内容作为上下文，让 Claude Code 根据仓库代码/配置回答。
    """
    repo_info = _get_repo_info(repo_dir)
    context = (
        f"msAgent 仓库已克隆到临时目录: {repo_dir}\n"
        f"文件数: {repo_info['file_count']}\n"
        f"顶级目录: {', '.join(repo_info['top_level'][:20])}\n\n"
        f"用户查询: {query}\n\n"
        f"请结合仓库中的代码和配置文件回答用户的问题。"
    )
    return await _run_skill("", context)


class MSAgentMCPServer:
    """msAgent MCP Server — 启动时拉取远程仓库，通过 stdio 提供 MCP 服务。

    协议: JSON-RPC 2.0 over stdin/stdout
    方法:
    - tools/list → 列出所有工具
    - tools/call → 执行指定工具
    - health/ping → 心跳
    """

    def __init__(self, repo_url: str = DEFAULT_REPO_URL):
        self._repo_url = repo_url
        self._repo_dir: Path | None = None
        self._tools: list[dict] = []
        self._initialized = False

    async def initialize(self) -> None:
        """克隆仓库并发现所有能力。"""
        tmp = Path(tempfile.mkdtemp(prefix="msagent-"))
        print(f"[msAgent] 正在从 {self._repo_url} 克隆...", file=sys.stderr)
        try:
            _clone_repo(self._repo_url, tmp / "repo")
            self._repo_dir = tmp
            repo_path = tmp / "repo"
            print(f"[msAgent] 克隆完成: {repo_path}", file=sys.stderr)

            # 发现能力
            skills = _discover_skills(repo_path)
            self._tools = []
            for s in skills:
                # 为每个 skill 暴露为一个 MCP tool
                self._tools.append({
                    "name": s["name"],
                    "description": s["description"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "传给 agent 的提示内容",
                            }
                        },
                        "required": ["prompt"],
                    },
                })

            # 总是添加通用查询工具
            self._tools.append({
                "name": "ms_agent_ask",
                "description": "向 msAgent 发送自然语言查询，结合仓库代码/配置回答",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要查询的问题",
                        }
                    },
                    "required": ["query"],
                },
            })

            # 状态检查工具
            self._tools.append({
                "name": "ms_agent_status",
                "description": "检查 msAgent 的状态信息（仓库版本、文件数等）",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            })

            self._initialized = True
            print(f"[msAgent] 已加载 {len(self._tools)} 个工具", file=sys.stderr)

        except Exception as e:
            print(f"[msAgent] 初始化失败: {e}", file=sys.stderr)
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    async def shutdown(self) -> None:
        """清理临时目录。"""
        if self._repo_dir:
            shutil.rmtree(self._repo_dir, ignore_errors=True)
            self._repo_dir = None
            print("[msAgent] 临时目录已清理", file=sys.stderr)

    async def _handle_request(self, request: dict) -> dict:
        req_id = request.get("id", 0)
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "tools/list":
            return self._handle_list_tools(req_id)
        elif method == "tools/call":
            return await self._handle_call_tool(req_id, params)
        elif method == "health/ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"status": "ok"}}
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"未知方法: {method}"},
            }

    def _handle_list_tools(self, req_id: int) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": self._tools},
        }

    async def _handle_call_tool(self, req_id: int, params: dict) -> dict:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if not tool_name:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32602, "message": "缺少 tool name"},
            }

        try:
            if tool_name == "ms_agent_status":
                if not self._repo_dir:
                    info_lines = ["[msAgent] 状态: 未初始化"]
                else:
                    info = _get_repo_info(self._repo_dir)
                    info_lines = [
                        "[msAgent] 状态: 已就绪",
                        f"  仓库: {self._repo_url}",
                        f"  临时目录: {self._repo_dir}",
                        f"  文件数: {info['file_count']}",
                        f"  仓库大小: {info['repo_size'] / 1024:.1f} KB",
                    ]
                    if info["top_level"]:
                        info_lines.append(f"  顶级内容: {', '.join(info['top_level'][:10])}")
                result_text = "\n".join(info_lines)

            elif tool_name == "ms_agent_ask":
                query = arguments.get("query", "")
                if not query:
                    return {
                        "jsonrpc": "2.0", "id": req_id,
                        "error": {"code": -32602, "message": "缺少 query 参数"},
                    }
                result_text = await _run_repo_agent(self._repo_dir, query)

            else:
                # 按 skill name 调用
                prompt = arguments.get("prompt", "")
                result_text = await _run_skill(tool_name, prompt)

            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                },
            }

        except Exception as e:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"[msAgent] 执行失败: {e}"}],
                    "isError": True,
                },
            }

    async def run_stdio(self) -> None:
        """通过 stdin/stdout 运行 MCP 协议主循环。"""
        await self.initialize()

        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        buffer = ""

        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        request = json.loads(line)
                    except json.JSONDecodeError as e:
                        error_resp = {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": f"JSON 解析错误: {e}"},
                        }
                        print(json.dumps(error_resp, ensure_ascii=False), flush=True)
                        continue

                    response = await self._handle_request(request)
                    print(json.dumps(response, ensure_ascii=False), flush=True)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[msAgent] 错误: {e}", file=sys.stderr)
        finally:
            await self.shutdown()

    def list_tools_text(self) -> str:
        """以可读文本形式列出所有工具。"""
        if not self._tools:
            raise RuntimeError("Server not initialized")
        lines = [f"msAgent MCP Server — 提供 {len(self._tools)} 个工具", ""]
        for t in self._tools:
            lines.append(f"  {t['name']}")
            lines.append(f"    描述: {t['description'][:80]}")
            lines.append("")
        return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="msAgent MCP Server (startup-clone mode)")
    parser.add_argument("--list-tools", action="store_true", help="列出所有已发现的 tools")
    parser.add_argument("--repo-url", type=str, default=DEFAULT_REPO_URL,
                        help="msAgent 仓库 URL")
    args = parser.parse_args()

    server = MSAgentMCPServer(repo_url=args.repo_url)

    if args.list_tools:
        asyncio.run(server.initialize())
        print(server.list_tools_text())
        asyncio.run(server.shutdown())
        return

    asyncio.run(server.run_stdio())


if __name__ == "__main__":
    main()
