"""CANNBot MCP Server — 启动时拉取仓库，将 SKILL.md 注册为 MCP 工具。

与旧版不同，不再在本地持久化 repo/skills：
  1. 启动时 git clone --depth=1 到临时目录
  2. 扫描所有 SKILL.md，解析 YAML frontmatter
  3. 每个 skill 注册为一个 MCP tool
  4. 关闭时清理临时目录

MCP 协议（stdio transport）:
    stdin/stdout 通过 JSON-RPC 2.0 通信
    - tools/list → 列出所有可用的 CANNBot skill
    - tools/call → 执行指定的 skill
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

DEFAULT_REPO_URL = "https://gitcode.com/cann/cannbot-skills.git"


def _clone_repo(url: str, dest: Path) -> None:
    """git clone --depth=1 到目标目录。"""
    result = subprocess.run(
        ["git", "clone", "--depth=1", url, str(dest)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"克隆 CANNBot skills 仓库失败: {result.stderr.strip() or result.stdout.strip()}"
        )


def _parse_skill_md(md_path: Path) -> dict[str, str] | None:
    """解析 SKILL.md 的 YAML frontmatter，提取 name 和 description。"""
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


def _discover_skills(repo_dir: Path) -> list[dict[str, str]]:
    """在克隆的仓库中递归查找所有 SKILL.md。"""
    seen: set[str] = set()
    skills: list[dict[str, str]] = []

    for skill_md in sorted(repo_dir.rglob("SKILL.md")):
        parsed = _parse_skill_md(skill_md)
        if parsed and parsed["name"] not in seen:
            seen.add(parsed["name"])
            skills.append({
                "name": parsed["name"],
                "description": parsed["description"],
                "skill_dir": str(skill_md.parent),
            })

    skills.sort(key=lambda s: s["name"])
    return skills


async def _invoke_skill(skill_name: str, prompt: str) -> str:
    """通过 Claude Code 调用 Skill。"""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from engine.claude_tools import stream_claude_skill

        chunks: list[str] = []
        async for chunk in stream_claude_skill(skill_name, prompt):
            if chunk:
                chunks.append(chunk)
        return "".join(chunks) if chunks else "(无输出)"
    except ImportError as e:
        return f"[CANNBot] 无法加载 claude-agent-sdk: {e}"
    except Exception as e:
        return f"[CANNBot] Skill 调用失败: {type(e).__name__}: {e}\n{traceback.format_exc()}"


class CANNBotMCPServer:
    """CANNBot MCP Server — 启动时拉取远程仓库，通过 stdio 提供 MCP 服务。

    协议: JSON-RPC 2.0 over stdin/stdout
    方法:
    - tools/list → 列出所有 skill
    - tools/call → 执行指定 skill
    - health/ping → 心跳
    """

    def __init__(self, repo_url: str = DEFAULT_REPO_URL):
        self._repo_url = repo_url
        self._repo_dir: Path | None = None
        self._skills: list[dict[str, str]] = []
        self._skills_map: dict[str, str] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """克隆仓库并发现所有 skills。"""
        # 克隆到临时目录
        tmp = Path(tempfile.mkdtemp(prefix="cannbot-"))
        print(f"[CANNBot] 正在从 {self._repo_url} 克隆...", file=sys.stderr)
        try:
            _clone_repo(self._repo_url, tmp / "repo")
            self._repo_dir = tmp
            repo_path = tmp / "repo"
            print(f"[CANNBot] 克隆完成: {repo_path}", file=sys.stderr)
        except Exception as e:
            print(f"[CANNBot] 克隆失败: {e}", file=sys.stderr)
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        # 发现 skills
        self._skills = _discover_skills(repo_path)
        self._skills_map = {s["name"]: s["skill_dir"] for s in self._skills}
        self._initialized = True

        if not self._skills:
            print("[CANNBot] 未发现任何 SKILL.md", file=sys.stderr)
        else:
            print(f"[CANNBot] 已加载 {len(self._skills)} 个 skills", file=sys.stderr)
            for s in self._skills:
                print(f"  - {s['name']}: {s['description'][:60]}", file=sys.stderr)

    async def shutdown(self) -> None:
        """清理临时目录。"""
        if self._repo_dir:
            shutil.rmtree(self._repo_dir, ignore_errors=True)
            self._repo_dir = None
            print("[CANNBot] 临时目录已清理", file=sys.stderr)

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
        tools = []
        for s in self._skills:
            tools.append({
                "name": s["name"],
                "description": s["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "传给 skill 的提示内容",
                        }
                    },
                    "required": ["prompt"],
                },
            })
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

    async def _handle_call_tool(self, req_id: int, params: dict) -> dict:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        prompt = arguments.get("prompt", "")

        if not tool_name:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32602, "message": "缺少 tool name"},
            }
        if tool_name not in self._skills_map:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32602, "message": f"未知 tool: {tool_name}"},
            }

        try:
            result_text = await _invoke_skill(tool_name, prompt)
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
                    "content": [{"type": "text", "text": f"[CANNBot] 执行失败: {e}"}],
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
            print(f"[CANNBot] 错误: {e}", file=sys.stderr)
        finally:
            await self.shutdown()

    def list_tools_text(self) -> str:
        """以可读文本形式列出所有工具（用于 --list-tools 模式）。"""
        if not self._skills:
            raise RuntimeError("Server not initialized")
        lines = [f"CANNBot MCP Server — 发现 {len(self._skills)} 个 skills", ""]
        for s in self._skills:
            lines.append(f"  {s['name']}")
            lines.append(f"    描述: {s['description'][:80]}")
            lines.append("")
        return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CANNBot MCP Server (startup-clone mode)")
    parser.add_argument("--list-tools", action="store_true", help="列出所有已发现的 tools")
    parser.add_argument("--repo-url", type=str, default=DEFAULT_REPO_URL,
                        help="CANNBot skills 仓库 URL")
    args = parser.parse_args()

    server = CANNBotMCPServer(repo_url=args.repo_url)

    if args.list_tools:
        asyncio.run(server.initialize())
        print(server.list_tools_text())
        asyncio.run(server.shutdown())
        return

    asyncio.run(server.run_stdio())


if __name__ == "__main__":
    main()
