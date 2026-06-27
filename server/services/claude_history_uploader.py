"""Claude Code 历史记录监控服务。

按固定间隔扫描 ~/.claude/projects/<project-dir> 下的 .jsonl 文件，
把进程运行期间新增的记录捞出来；上传逻辑由调用方后续注入。

并发写入处理：Claude Code 同步追加写 .jsonl，可能在写入一半时被读取。
本模块采用"原子快照 + 仅处理完整行"策略：
1. 一次性读取整个文件内容；
2. 以换行符分割，丢弃未以 \\n 结尾的最后一行（写入中的半行）；
3. 只把完整行计入偏移量并解析上传，避免半行污染或丢失数据。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from server.config import settings
from server.services.claude_history_extractor import HistoryExtractor, Turn
from lts_logger import lts_logger


def get_claude_projects_root() -> Path:
    """返回 Claude Code projects 根目录。"""
    return Path.home() / ".claude" / "projects"


def _derive_project_dir_name(cwd: Path) -> str:
    """根据工作目录推导 Claude Code 项目目录名（Linux 环境）。

    示例：/home/user/mofix-new -> home-user-mofix-new
    """
    posix = cwd.as_posix().rstrip("/")
    if posix.startswith("/"):
        posix = posix[1:]
    return posix.replace("/", "-")


def get_watched_project_dirs() -> list[Path]:
    """定位需要监控的 Claude Code history 目录列表。

    如果配置了 claude_history_project_dir，则只监控该目录；
    否则监控 ~/.claude/projects 下所有子目录。
    """
    root = get_claude_projects_root()
    if settings.claude_history_project_dir:
        return [root / settings.claude_history_project_dir]
    if not root.exists():
        return []
    return sorted([d for d in root.iterdir() if d.is_dir()])


def _read_complete_lines(file: Path) -> tuple[list[str], int]:
    """原子读取文件，返回完整行列表与完整行数。

    未以换行符结尾的最后一行视为写入中的半行，不纳入统计和返回。
    """
    try:
        content = file.read_text(encoding="utf-8")
    except Exception as e:
        lts_logger.log(
            "claude_history",
            level="error",
            message="读取历史记录文件失败",
            file=str(file),
            error=str(e),
        )
        return [], 0

    if not content:
        return [], 0

    # split 后最后一项一定是空串（内容以 \n 结尾）或不完整行（未以 \n 结尾）
    parts = content.split("\n")
    complete_lines = parts[:-1]
    return complete_lines, len(complete_lines)


class HistoryWatcher:
    """监控 Claude Code 历史记录目录并捞出新增记录。"""

    def __init__(self, interval: float = 60.0) -> None:
        self._interval = interval
        self._offsets: dict[Path, int] = {}
        self._stop_event = asyncio.Event()
        # 每个 session 文件独立提炼，避免多项目/多会话记录互相干扰
        self._extractors: dict[Path, HistoryExtractor] = {}

    async def start(self) -> None:
        """启动轮询循环。"""
        lts_logger.log(
            "claude_history",
            level="info",
            message="启动 Claude Code 历史记录监控",
            project_dirs=[str(d) for d in get_watched_project_dirs()],
            interval=self._interval,
        )

        while not self._stop_event.is_set():
            try:
                await self._scan_once()
            except Exception as e:
                lts_logger.log(
                    "claude_history",
                    level="error",
                    message="历史记录扫描异常",
                    error=str(e),
                )

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    async def stop(self) -> None:
        """停止轮询循环。"""
        self._stop_event.set()

    async def _scan_once(self) -> None:
        """执行一次扫描。"""
        project_dirs = get_watched_project_dirs()
        if not project_dirs:
            lts_logger.log(
                "claude_history",
                level="debug",
                message="没有可监控的历史记录目录",
            )
            return

        for project_dir in project_dirs:
            if not project_dir.exists():
                lts_logger.log(
                    "claude_history",
                    level="debug",
                    message="历史记录目录不存在，跳过",
                    project_dir=str(project_dir),
                )
                continue
            await self._scan_project_dir(project_dir)

    async def _scan_project_dir(self, project_dir: Path) -> None:
        """扫描单个项目目录下的所有 jsonl 文件。"""
        jsonl_files = sorted(project_dir.glob("*.jsonl"))
        for file in jsonl_files:
            complete_lines, current_lines = await asyncio.to_thread(_read_complete_lines, file)

            if file not in self._offsets:
                # 首次见到该文件：
                # 1. 用 extractor 预演已有完整行，使未完成的 human 提问进入 pending，
                #    这样后续新增的 assistant 回复能找到所属轮次；
                # 2. 丢弃已完成的轮次（遵守"启动时已存在记录不关注"）；
                # 3. 记录偏移量，之后只处理新增行。
                existing_records = self._parse_records(complete_lines)
                if existing_records:
                    extractor = self._extractors.setdefault(file, HistoryExtractor())
                    # 忽略已完成的旧轮次，仅保留 pending
                    _ = extractor.extract_turns(existing_records)
                self._offsets[file] = current_lines
                lts_logger.log(
                    "claude_history",
                    level="debug",
                    message="首次发现历史记录文件，记录偏移量",
                    project_dir=project_dir.name,
                    file=file.name,
                    offset=current_lines,
                )
                continue

            offset = self._offsets[file]
            if current_lines < offset:
                # 文件被截断或删除，重置偏移量
                lts_logger.log(
                    "claude_history",
                    level="warning",
                    message="历史记录文件行数减少，重置偏移量",
                    project_dir=project_dir.name,
                    file=file.name,
                    old_offset=offset,
                    new_offset=current_lines,
                )
                self._offsets[file] = current_lines
                continue

            if current_lines == offset:
                continue

            new_records = self._parse_records(complete_lines[offset:])
            self._offsets[file] = current_lines

            if new_records:
                extractor = self._extractors.setdefault(file, HistoryExtractor())
                completed_turns = extractor.extract_turns(new_records)
                for turn in completed_turns:
                    await self._upload_turn(file, turn)

    def _parse_records(self, lines: list[str]) -> list[dict[str, Any]]:
        """把完整行解析为 JSON 记录。"""
        records: list[dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                lts_logger.log(
                    "claude_history",
                    level="warning",
                    message="历史记录 JSON 解析失败，已跳过该行",
                    line=line,
                )
        return records

    async def _upload_turn(self, file: Path, turn: Turn) -> None:
        """上传单个提炼后的对话轮次（占位实现，后续接入外部接口）。

        Args:
            file: 来源 jsonl 文件。
            turn: 已提炼的对话轮次。
        """
        print(
            f"[claude-history] turn completed: {file.name} "
            f"(session_id={file.stem}, turn_id={turn.turn_id}, "
            f"is_complete={turn.is_complete})",
            flush=True,
        )
        print(
            f"[claude-history] turn payload: "
            f"{json.dumps(turn.to_dict(), ensure_ascii=False)}",
            flush=True,
        )
