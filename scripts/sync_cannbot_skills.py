#!/usr/bin/env python3
"""CANNBot SKILL.md 同步脚本（静态模式）。

注意: 推荐方式已改为 MCP Server 启动时自动从远程仓库克隆并注册为 MCP 工具
（参见 app/tools/mcp_servers/cannbot_server.py），不再需要本地持久化。

此脚本保留用于离线评估/调试，生产环境不需要运行。"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("sync_cannbot_skills")

# ── 默认配置 ─────────────────────────────────────────────────────────────

DEFAULT_REPO_URL = "https://gitcode.com/cann/cannbot-skills.git"
# 默认输出到 repo 内的 agents/cannbot/ 目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "agents" / "cannbot"
DEFAULT_DAEMON_INTERVAL = 360  # 分钟


# ── 核心逻辑 ─────────────────────────────────────────────────────────────

class CannbotSkillSync:
    """CANNBot skills 同步器。"""

    def __init__(self, output_dir: Path, repo_url: str = DEFAULT_REPO_URL):
        self._output_dir = output_dir.resolve()
        self._repo_dir = self._output_dir / "repo"
        self._skills_dir = self._output_dir / "skills"
        self._repo_url = repo_url

    def sync(self) -> int:
        """执行同步，返回同步的 skill 数量。"""
        self._ensure_dirs()
        self._clone_or_pull()
        count = self._copy_skills()
        self._cleanup_stale(count)
        self._print_summary(count)
        return count

    def _ensure_dirs(self):
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._skills_dir.mkdir(parents=True, exist_ok=True)

    def _clone_or_pull(self):
        """克隆或拉取最新代码。"""
        if self._repo_dir.is_dir():
            # 已有仓库，拉取更新
            logger.info("拉取最新更新...")
            result = subprocess.run(
                ["git", "-C", str(self._repo_dir), "pull", "--ff-only"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.warning(f"git pull 失败: {result.stderr.strip()}")
                git_fetch = subprocess.run(
                    ["git", "-C", str(self._repo_dir), "fetch", "origin"],
                    capture_output=True, text=True, timeout=120,
                )
                if git_fetch.returncode == 0:
                    subprocess.run(
                        ["git", "-C", str(self._repo_dir), "reset", "--hard", "origin/main"],
                        capture_output=True, text=True, timeout=60,
                    )
                    logger.info("已强制重置到 origin/main")
                else:
                    logger.warning("git fetch 也失败，将继续使用本地缓存")
            else:
                logger.info(f"已更新: {result.stdout.strip()}")
        else:
            # 克隆仓库
            logger.info(f"从 {self._repo_url} 克隆...")
            result = subprocess.run(
                ["git", "clone", "--depth=1", self._repo_url, str(self._repo_dir)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"克隆失败: {result.stderr.strip() or result.stdout.strip()}"
                )
            logger.info("克隆完成")

    def _copy_skills(self) -> int:
        """从仓库复制所有 SKILL.md 到 skills 目录。"""
        skill_dirs = self._find_skill_dirs()
        count = 0
        for src_dir in skill_dirs:
            skill_name = src_dir.name
            target = self._skills_dir / skill_name
            target.mkdir(parents=True, exist_ok=True)

            # 复制 SKILL.md
            shutil.copy2(src_dir / "SKILL.md", target / "SKILL.md")

            # 复制 references 子目录（如果存在）
            ref_src = src_dir / "references"
            if ref_src.is_dir():
                ref_target = target / "references"
                if ref_target.is_dir():
                    shutil.rmtree(ref_target)
                shutil.copytree(ref_src, ref_target)

            count += 1
            logger.debug(f"  + {skill_name}")

        return count

    def _find_skill_dirs(self) -> list[Path]:
        """在仓库中查找包含 SKILL.md 的目录（最多深入 3 层）。"""
        results: list[Path] = []
        if not self._repo_dir.is_dir():
            return results

        # 扫描根目录级别
        for item in sorted(self._repo_dir.iterdir()):
            if item.is_dir() and (item / "SKILL.md").exists():
                results.append(item)

        # 扫描二级目录
        for item in sorted(self._repo_dir.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                for sub in sorted(item.iterdir()):
                    if sub.is_dir() and (sub / "SKILL.md").exists():
                        results.append(sub)

        return results

    def _cleanup_stale(self, current_count: int):
        """清理已不存在的 skill 目录。"""
        # 获取当前同步的 skill 名称
        active_skills = set()
        for skill_dir in self._skills_dir.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                active_skills.add(skill_dir.name)

        if not active_skills:
            return

    def _print_summary(self, count: int):
        """输出同步摘要。"""
        sync_paths = [str(self._repo_dir), str(self._skills_dir)]
        logger.info(f"同步完成: {count} 个 skills")
        logger.info(f"  仓库: {sync_paths[0]}")
        logger.info(f"  Skills: {sync_paths[1]}")
        logger.info(f"  设置 CANNBOT_SKILLS_DIR={self._skills_dir} 使服务器加载")


# ── 命令行 ────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_daemon(sync: CannbotSkillSync, interval_minutes: int):
    """以守护进程模式运行，定期同步。"""
    logger.info(f"守护进程启动，同步间隔: {interval_minutes} 分钟")
    while True:
        try:
            sync.sync()
        except Exception as e:
            logger.error(f"同步失败: {e}")
        logger.info(f"等待 {interval_minutes} 分钟后下一次同步...")
        time.sleep(interval_minutes * 60)


def main():
    parser = argparse.ArgumentParser(
        description="CANNBot SKILL.md 自动同步工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--once", action="store_true",
        help="执行一次同步后退出",
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="以守护进程模式运行",
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_DAEMON_INTERVAL,
        help=f"守护进程同步间隔（分钟），默认 {DEFAULT_DAEMON_INTERVAL}",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"输出目录，默认 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--repo-url", type=str, default=DEFAULT_REPO_URL,
        help="CANNBot skills 仓库 URL",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细输出",
    )

    args = parser.parse_args()

    if not args.once and not args.daemon:
        # 默认行为：执行一次
        args.once = True

    setup_logging(args.verbose)
    output_dir = Path(args.output_dir).expanduser()
    sync = CannbotSkillSync(output_dir=output_dir, repo_url=args.repo_url)

    if args.once:
        count = sync.sync()
        sys.exit(0 if count > 0 else 1)
    elif args.daemon:
        run_daemon(sync, args.interval)


if __name__ == "__main__":
    main()
