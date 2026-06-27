#!/usr/bin/env python3
"""从 gitcode.com/Ascend/model-agent 同步并分级 Skills。

工作流:
  1. git clone --depth=1 或 git pull 拉取最新 model-agent 仓库
  2. 递归发现所有 SKILL.md
  3. 调用 evaluate_skills.py 进行 11 维评分
  4. 按 tier 过滤（T1≥80, T2≥65, T3≥50）
  5. 复制到 agents/model-agent/skills/
  6. 生成 agents/model-agent/tiers.json
  7. 生成 agents/model-agent/INDEX.md

用法:
    python scripts/sync_model_agent_skills.py                  # 完整同步
    python scripts/sync_model_agent_skills.py --clone          # 强制重新 clone
    python scripts/sync_model_agent_skills.py --min-tier B     # 只同步 ≥B 级 (默认 A)
    python scripts/sync_model_agent_skills.py --dry-run        # 试运行，不写文件
    python scripts/sync_model_agent_skills.py --skip-eval      # 跳过评分（复用已有结果）
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ── 路径 ─────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_AGENT_DIR = _PROJECT_ROOT / "agents" / "model-agent"
_SKILLS_DIR = _AGENT_DIR / "skills"
_TIERS_JSON = _AGENT_DIR / "tiers.json"
_INDEX_MD = _AGENT_DIR / "INDEX.md"

MODEL_AGENT_REPO = "https://gitcode.com/Ascend/model-agent.git"
MODEL_AGENT_BRANCH = "master"

# 从 evaluate_skills.py 导入评分函数
sys.path.insert(0, str(_PROJECT_ROOT))
from scripts.evaluate_skills import (  # noqa: E402
    DEFAULT_TIER1_SKILLS,
    batch_evaluate,
    build_tiers,
    find_all_skill_mds,
    parse_skill_md,
    calculate_score,
)


# ── 仓库管理 ────────────────────────────────────────────────────────

def _clone_repo(tmp_dir: Path, force: bool = False) -> Path | None:
    """将 model-agent 仓库克隆到临时目录。"""
    dest = tmp_dir / "model-agent"
    if dest.exists():
        print(f"[sync] 目录已存在，执行 git pull...")
        try:
            subprocess.run(
                ["git", "pull", "origin", MODEL_AGENT_BRANCH],
                cwd=dest,
                capture_output=True,
                timeout=120,
            )
            return dest
        except Exception as e:
            print(f"[sync] git pull 失败: {e}，重新 clone...")
            shutil.rmtree(dest)

    print(f"[sync] 克隆 {MODEL_AGENT_REPO} (--depth=1)...")
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", "-b", MODEL_AGENT_BRANCH,
             MODEL_AGENT_REPO, str(dest)],
            capture_output=True,
            timeout=300,
            check=True,
        )
        print(f"[sync] 克隆完成: {dest}")
        return dest
    except subprocess.CalledProcessError as e:
        print(f"[sync] 克隆失败: {e.stderr.decode() if e.stderr else e}")
        return None
    except Exception as e:
        print(f"[sync] 克隆异常: {e}")
        return None


# ── 评分与过滤 ──────────────────────────────────────────────────────

def _run_evaluation(repo_dir: Path) -> list[dict[str, Any]]:
    """对仓库下所有 SKILL.md 进行 11 维评分。"""
    skill_mds = find_all_skill_mds(repo_dir)
    print(f"[sync] 发现 {len(skill_mds)} 个 SKILL.md 文件")
    results = batch_evaluate(skill_mds)
    print(f"[sync] 评分完成: {len(results)} 个 skills")
    return results


def _filter_by_tier(results: list[dict[str, Any]], min_tier: int) -> dict[int, list]:
    """按最低 tier 过滤评分结果。"""
    by_tier: dict[int, list] = {1: [], 2: [], 3: []}
    for r in results:
        t = r["tier"]
        if t == 0:
            continue  # tier 0 = 不收录
        if t >= min_tier:
            by_tier[t].append(r)
    return by_tier


# ── 文件同步 ────────────────────────────────────────────────────────

def _sync_skill_files(results: list[dict[str, Any]], dry_run: bool) -> int:
    """将评分达标的 skills 复制到 agents/model-agent/skills/。"""
    if not _SKILLS_DIR.exists() and not dry_run:
        _SKILLS_DIR.mkdir(parents=True)
        print(f"[sync] 创建目录: {_SKILLS_DIR}")

    copied = 0
    for r in results:
        src = Path(r.get("file_path", ""))
        if not src or not src.exists():
            continue
        name = r["name"]
        dst = _SKILLS_DIR / name
        dst_skill_md = dst / "SKILL.md"

        if dst_skill_md.exists():
            continue  # 已存在，跳过

        if dry_run:
            print(f"  [DRY-RUN] cp {src} → {dst_skill_md}")
            copied += 1
            continue

        dst.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(src), str(dst_skill_md))
            # 复制关联的 references/ assets/ scripts/ 目录
            parent_dir = src.parent
            for subdir in ["references", "assets", "scripts"]:
                src_sub = parent_dir / subdir
                if src_sub.is_dir():
                    dst_sub = dst / subdir
                    if not dst_sub.exists():
                        shutil.copytree(str(src_sub), str(dst_sub),
                                        ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))
            copied += 1
        except Exception as e:
            print(f"  [sync] 复制 {name} 失败: {e}", file=sys.stderr)

    return copied


# ── INDEX.md 生成 ───────────────────────────────────────────────────

def _generate_index(tiers: dict[str, dict], results: list[dict]) -> str:
    """生成 INDEX.md。"""
    lines = [
        "# Model-Agent Skills 索引",
        "",
        f"> 更新于: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 总计: {len(results)} 个 skills | "
        f"Tier 1: {len(tiers.get('tier1', {}))} | "
        f"Tier 2: {len(tiers.get('tier2', {}))} | "
        f"Tier 3: {len(tiers.get('tier3', {}))}",
        "",
    ]

    for tier_label, tier_key, badge in [
        ("Tier 1 — 核心技能（≥80 分）", "tier1", "⭐⭐⭐"),
        ("Tier 2 — 扩展技能（65-79 分）", "tier2", "⭐⭐"),
        ("Tier 3 — 长尾技能（50-64 分）", "tier3", "⭐"),
    ]:
        skills = tiers.get(tier_key, {})
        if not skills:
            continue
        lines.append(f"## {tier_label} {badge}")
        lines.append("")
        lines.append("| 名称 | 评分 | 描述 |")
        lines.append("|------|:----:|------|")
        for name in sorted(skills.keys()):
            info = skills[name]
            desc = info.get("description", "")[:80]
            score = info.get("score", "?")
            lines.append(f"| {name} | {score} | {desc} |")
        lines.append("")

    # 按分类统计
    lines.append("## 分类统计")
    lines.append("")
    lines.append("| 分类 | 数量 |")
    lines.append("|------|:----:|")
    categories: dict[str, int] = {}
    for r in results:
        fp = r.get("file_path", "")
        # 从文件路径推断分类：skills/adaptation/xxx/SKILL.md → adaptation
        parts = Path(fp).parts if fp else []
        for i, p in enumerate(parts):
            if p in ("skills",) and i + 1 < len(parts):
                cat = parts[i + 1]
                categories[cat] = categories.get(cat, 0) + 1
                break
    for cat in sorted(categories.keys()):
        lines.append(f"| {cat} | {categories[cat]} |")
    lines.append("")

    return "\n".join(lines)


# ── 主流程 ──────────────────────────────────────────────────────────

def sync(
    clone: bool = False,
    min_tier: int = 3,
    dry_run: bool = False,
    skip_eval: bool = False,
) -> int:
    """执行完整同步流程。

    Returns:
        同步的 skill 数量。
    """
    # Step 1: 获取 model-agent 仓库
    if clone:
        tmp_dir = Path(tempfile.mkdtemp(prefix="model-agent-sync-"))
        repo_dir = _clone_repo(tmp_dir, force=clone)
        if not repo_dir:
            print("[sync] 仓库获取失败", file=sys.stderr)
            return -1
    else:
        # 检查是否已经有本地缓存
        if _SKILLS_DIR.is_dir() and any(_SKILLS_DIR.iterdir()) and not dry_run:
            print(f"[sync] 检测到已有技能目录: {_SKILLS_DIR}")
            print(f"[sync] 使用 --clone 重新同步")
            return _reindex_local(min_tier, dry_run)
        # 没有则 clone
        tmp_dir = Path(tempfile.mkdtemp(prefix="model-agent-sync-"))
        repo_dir = _clone_repo(tmp_dir, force=True)
        if not repo_dir:
            print("[sync] 仓库获取失败", file=sys.stderr)
            return -1

    # Step 2: 评分
    if skip_eval:
        try:
            tiers_data = json.loads(_TIERS_JSON.read_text(encoding="utf-8"))
            print(f"[sync] 从 {_TIERS_JSON} 读取已有评分")
            results = []
            for tier_key in ["tier1", "tier2", "tier3"]:
                for name, info in tiers_data.get(tier_key, {}).items():
                    results.append({
                        "name": name,
                        "description": info.get("description", ""),
                        "total": info.get("score", 0),
                        "tier": int(tier_key[-1]),
                        "file_path": info.get("file_path", ""),
                    })
        except Exception as e:
            print(f"[sync] 加载已有评分失败: {e}，重新执行评分")
            results = _run_evaluation(repo_dir)
    else:
        results = _run_evaluation(repo_dir)

    # Step 3: 过滤
    by_tier = _filter_by_tier(results, min_tier=min_tier)
    selected = by_tier[1] + by_tier[2] + by_tier[3]
    print(f"[sync] 过滤后: T1={len(by_tier[1])}  T2={len(by_tier[2])}  T3={len(by_tier[3])}")

    # Step 4: 构建分级
    tiers = build_tiers(selected)

    # Step 5: 复制文件
    copied = _sync_skill_files(selected, dry_run)
    print(f"[sync] 复制 {copied} 个 skills 到 {_SKILLS_DIR}")

    if dry_run:
        print(f"[sync] DRY-RUN 模式 — 未写入任何文件")
        return copied

    # Step 6: 写 tiers.json
    _TIERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    _TIERS_JSON.write_text(
        json.dumps(tiers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[sync] 写入 {_TIERS_JSON}")

    # Step 7: 写 INDEX.md
    index_content = _generate_index(tiers, selected)
    _INDEX_MD.write_text(index_content, encoding="utf-8")
    print(f"[sync] 写入 {_INDEX_MD}")

    # 清理临时目录
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    print(f"[sync] 同步完成: {copied} 个 skills (T1={len(by_tier[1])} T2={len(by_tier[2])} T3={len(by_tier[3])})")
    return copied


def _reindex_local(min_tier: int, dry_run: bool) -> int:
    """对本地已有 skills 重新评分建索引（不重新 clone）。"""
    skill_mds = find_all_skill_mds(_SKILLS_DIR)
    if not skill_mds:
        print(f"[sync] {_SKILLS_DIR} 中未找到 SKILL.md")
        return -1

    print(f"[sync] 对 {len(skill_mds)} 个本地 skills 重新评分...")
    results = batch_evaluate(skill_mds)
    by_tier = _filter_by_tier(results, min_tier)
    selected = by_tier[1] + by_tier[2] + by_tier[3]
    tiers = build_tiers(selected)

    if dry_run:
        print(f"[sync] DRY-RUN: T1={len(by_tier[1])} T2={len(by_tier[2])} T3={len(by_tier[3])}")
        return len(selected)

    _TIERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    _TIERS_JSON.write_text(
        json.dumps(tiers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _INDEX_MD.write_text(_generate_index(tiers, selected), encoding="utf-8")
    print(f"[sync] 重新索引完成: T1={len(by_tier[1])} T2={len(by_tier[2])} T3={len(by_tier[3])}")
    return len(selected)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="同步 Model-Agent Skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--clone", action="store_true", help="强制重新 clone 仓库")
    parser.add_argument("--min-tier", type=str, default="C",
                        choices=["S", "A", "B", "C"],
                        help="最低 tier: S(≥80) A(≥65) B(≥50) C(≥0) (默认 C)")
    parser.add_argument("--dry-run", action="store_true", help="试运行，不写文件")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评分（复用已有结果）")
    parser.add_argument("--reindex", action="store_true", help="对本地 skills 重新评分建索引")

    args = parser.parse_args()
    tier_map = {"S": 1, "A": 2, "B": 3, "C": 0}

    if args.reindex:
        count = _reindex_local(tier_map[args.min_tier], args.dry_run)
    else:
        count = sync(
            clone=args.clone,
            min_tier=tier_map[args.min_tier],
            dry_run=args.dry_run,
            skip_eval=args.skip_eval,
        )

    if count < 0:
        sys.exit(1)
    print(f"\n完成！共处理 {count} 个 skills。")


if __name__ == "__main__":
    main()
