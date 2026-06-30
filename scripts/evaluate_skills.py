#!/usr/bin/env python3
"""Skills 11维评价工具 — 对 SKILL.md 进行结构化评分和 Tier 分级。

评价维度:
  结构维度（50分）: frontmatter质量(5) + 工作流清晰度(10) + 边界条件(8)
                    + 检查点设计(5) + 指令具体性(12) + 资源整合(5) + 兼容性(5)
  业务维度（50分）: 业务相关性(12) + 不可替代性(12) + 可执行性(10) + 影响力(10) + 成熟度(6)

Tier 划分:
  Tier 1 (核心): ≥ 80 分 — 始终注入 prompt
  Tier 2 (扩展): ≥ 65 分 — 名称列表进 prompt，引用后展开
  Tier 3 (长尾): ≥ 50 分 — 不进 prompt，/skill-search 可查
  Below: < 50 分 — 暂不收录

用法:
    # 评估单个 SKILL.md
    python scripts/evaluate_skills.py path/to/SKILL.md

    # 批量评估目录下所有 SKILL.md
    python scripts/evaluate_skills.py --batch skills/ascend/

    # 输出 JSON 报告
    python scripts/evaluate_skills.py --batch skills/ascend/ --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


# ── 评分维度定义 ─────────────────────────────────────────────────────────

# 标准 SKILL.md 名称（v0.3.0 内置），这些直接进入 Tier 1
DEFAULT_TIER1_SKILLS: set[str] = {
    "adapt-agent", "verify-agent", "optimizer-agent", "quantify-agent",
    "ai4s-main", "ai4s-basic", "ai4s-perf-tuning", "ai4s-precision-alignment",
    "ai4s-profiling", "ascend-optimization", "ascend-affinity-operator",
    "ascend-history-to-skill", "gitcode-publish",
}


# ── SKILL.md 解析 ───────────────────────────────────────────────────────

def parse_skill_md(md_path: Path) -> dict[str, str] | None:
    """解析 SKILL.md 的 YAML frontmatter，提取元数据。"""
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
    meta: dict[str, str] = {"name": "", "description": "", "keywords": ""}
    in_desc = False
    desc_lines: list[str] = []
    body_text = text[end + 3:].strip()

    for line in front.splitlines():
        stripped = line.strip()
        if stripped.startswith("name:"):
            meta["name"] = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            in_desc = False
        elif stripped.startswith("description:"):
            raw = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            desc_lines = [raw]
            in_desc = True
        elif in_desc and (stripped.startswith("-") or stripped.startswith(">")
                          or stripped.startswith("https://") or stripped.startswith("http://")):
            desc_lines.append(stripped.lstrip("- >").strip())
        elif in_desc and stripped:
            in_desc = False
        elif stripped.startswith("keywords:") or stripped.startswith("keyword:"):
            meta["keywords"] = stripped.split(":", 1)[1].strip()

    meta["description"] = " ".join(desc_lines) if desc_lines else ""
    meta["body_length"] = str(len(body_text))
    meta["has_code_block"] = "true" if "```" in body_text else "false"

    return meta


# ── 结构维度评分（自动）─────────────────────────────────────────────────

def score_frontmatter(meta: dict[str, str]) -> int:
    """维度1: Frontmatter 质量 (0-5)"""
    score = 0
    name = meta.get("name", "")
    desc = meta.get("description", "")
    if name and re.match(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", name):
        score += 2  # 名称合规
    if desc and len(desc) > 20:
        score += 2  # 有意义的描述
    if "当" in desc or "when" in desc.lower() or "适用于" in desc:
        score += 1  # 包含触发条件
    return min(score, 5)


def score_workflow(meta: dict[str, str], body: str) -> int:
    """维度2: 工作流清晰度 (0-10)"""
    score = 0
    # 有序号步骤（1. 2. 3. 或 - [ ] 格式）
    steps = re.findall(r"(?:^|\n)\s*(?:[1-9]\d*\.\s|[-*]\s+\[.?\]\s|步骤\s*[1-9])", body)
    score += min(len(steps), 5) * 2  # 每个步骤 2 分，最多 10 分

    # 有流程图或伪代码
    if "```" in body:
        score += 2
    # 有输入输出描述
    if re.search(r"(?:输入|输出|input|output|参数|param)", body, re.IGNORECASE):
        score += 2
    return min(score, 10)


def score_boundary(meta: dict[str, str], body: str) -> int:
    """维度3: 边界条件覆盖 (0-8)"""
    score = 0
    keywords = [
        r"异常|错误|error|exception|fail",
        r"fallback|回退|降级",
        r"超时|timeout",
        r"限流|rate.limit|限制",
        r"不支持|not.supported|限制",
        r"如果.*失败|if.*fail|当.*不",
        r"取消|中断|cancel|interrupt",
        r"重试|retry|重试",
    ]
    for kw in keywords:
        if re.search(kw, body, re.IGNORECASE):
            score += 1
    return min(score, 8)


def score_checkpoint(meta: dict[str, str], body: str) -> int:
    """维度4: 检查点设计 (0-5)"""
    score = 0
    keywords = [
        r"确认|confirm|approve|批准",
        r"检查|check|verify|验证",
        r"用户确认|human.in.the.loop|人工",
        r"审批|review|审查",
    ]
    for kw in keywords:
        if re.search(kw, body, re.IGNORECASE):
            score += 1
    # 至少有 2 个以上检查点关键词 → 加分
    if score >= 2:
        score += 1
    return min(score, 5)


def score_specificity(meta: dict[str, str], body: str) -> int:
    """维度5: 指令具体性 (0-12)"""
    score = 0
    # 有代码示例
    code_blocks = body.count("```")
    score += min(code_blocks, 4) * 2  # 最多 8 分

    # 有具体命令或 API 调用
    if re.search(r"(?:python|bash|docker|pip|git|npu-smi|ascendc)\s", body):
        score += 2
    # 有配置文件或参数表
    if "|" in body and re.search(r"\|.*\|.*\|", body):
        score += 2  # Markdown 表格
    return min(score, 12)


def score_resources(meta: dict[str, str], skill_dir: Path) -> int:
    """维度6: 资源整合度 (0-5)"""
    score = 0
    if not skill_dir.is_dir():
        return 0
    # references/ 目录
    ref_dir = skill_dir / "references"
    if ref_dir.is_dir():
        score += 2
    # assets/ 目录
    if (skill_dir / "assets").is_dir():
        score += 1
    # scripts/ 目录
    if (skill_dir / "scripts").is_dir():
        score += 2
    return min(score, 5)


def score_compatibility(meta: dict[str, str]) -> int:
    """维度7: 兼容性 — 与现有 skills 不冲突 (0-5)"""
    name = meta.get("name", "")
    # 检查是否与已知 CANNBot skills 命名冲突
    known_prefixes = {"ascendc-", "triton-", "model-infer-", "pypto-",
                      "catlass-", "tilelang-", "torch-", "ops-", "gitcode-"}
    for prefix in known_prefixes:
        if name.startswith(prefix):
            return 3  # 可能有重叠，给中等分数
    if not name:
        return 0
    return 5  # 全新命名空间，完全兼容


# ── 业务维度评分（基于关键词和规则的启发式评分）────────────────────────

def score_business_relevance(meta: dict[str, str]) -> int:
    """维度8: 业务相关性 (0-12) — 是否解决真实 NPU 适配/调优问题"""
    desc = (meta.get("description", "") + " " + meta.get("name", "") + " " + meta.get("keywords", "")).lower()
    high = ["migrat", "adapt", "deploy", "infer", "optimize", "optim",
            "ascend", "npu", "cann", "vllm", "megatron"]
    medium = ["profiling", "profil", "convert", "quant", "precision",
              "test", "coverage", "docker", "install", "driver"]
    low = ["doc", "search", "generat", "qa", "faq", "audit"]

    score = 0
    for kw in high:
        if kw in desc:
            score += 3
    for kw in medium:
        if kw in desc:
            score += 2
    for kw in low:
        if kw in desc:
            score += 1
    return min(score, 12)


def score_uniqueness(meta: dict[str, str]) -> int:
    """维度9: 不可替代性 (0-12) — 是否提供现有 skills 没有的能力"""
    name = meta.get("name", "")
    # 已有 CANNBot 覆盖的前缀
    existing_prefixes = {"ascendc-", "triton-", "model-infer-", "pypto-",
                         "catlass-", "tilelang-", "torch-", "ops-", "gitcode-"}
    for prefix in existing_prefixes:
        if name.startswith(prefix):
            return 4  # 有重叠，不可替代性低
    # 模型名部署类（特定模型部署）→ 中等
    if any(m in name for m in ["npu", "deploy", "adapt"]):
        return 7
    # 全新的概念 → 高
    new_concepts = ["megatron", "mindspeed", "atc", "coverage", "pytest",
                    "unittest", "docker", "fsdp2", "code-comprehension",
                    "auto-bug", "skill-auditor"]
    desc_lower = meta.get("description", "").lower()
    for c in new_concepts:
        if c in desc_lower or c in name:
            return 11
    return 6


def score_executability(meta: dict[str, str]) -> int:
    """维度10: 可执行性 (0-10) — 在 DR-A3 上能否直接运行"""
    desc = (meta.get("description", "") + " " + meta.get("name", "")).lower()
    # 纯 Python/SKILL.md → 高
    pure_skills = ["search", "docs", "review", "audit", "generat", "design"]
    for kw in pure_skills:
        if kw in desc:
            return 9
    # 需要 docker → 中等（DR-A3 已有 docker）
    if "docker" in desc:
        return 8
    # 需要 CANN/npu-smi → 高（DR-A3 已有）
    if "npu" in desc or "ascend" in desc or "cann" in desc or "vllm" in desc:
        return 8
    # 需要特定硬件或数据集 → 低
    if "download" in desc and ("model" in desc or "weight" in desc):
        return 5
    return 7


def score_impact(meta: dict[str, str]) -> int:
    """维度11: 用户影响力 (0-10) — 覆盖的用户场景广度"""
    desc = (meta.get("description", "") + " " + meta.get("name", "")).lower()
    high_impact = ["migrat", "adapt", "deploy", "infer", "verif", "test",
                   "optim", "profil", "convert"]
    for kw in high_impact:
        if kw in desc:
            return 9
    mid_impact = ["doc", "search", "generat", "install", "config", "env"]
    for kw in mid_impact:
        if kw in desc:
            return 7
    return 5


def score_maturity(meta: dict[str, str]) -> int:
    """维度12: 成熟度 (0-6) — SKILL.md 文档完整度"""
    score = 0
    desc = meta.get("description", "")
    body_len = int(meta.get("body_length", "0"))
    # 描述长度
    if len(desc) > 100:
        score += 2
    elif len(desc) > 50:
        score += 1
    # 正文长度（越详细越成熟）
    if body_len > 5000:
        score += 2
    elif body_len > 2000:
        score += 1
    # 有代码块
    if meta.get("has_code_block") == "true":
        score += 2
    return min(score, 6)


# ── 总分计算与 Tier 划分 ──────────────────────────────────────────────

def calculate_score(meta: dict[str, str], body: str, skill_dir: Path) -> dict[str, Any]:
    """执行 11 维评分，返回完整评分报告。"""
    name = meta.get("name", "")

    # 结构维度（50分）
    s1 = score_frontmatter(meta)
    s2 = score_workflow(meta, body)
    s3 = score_boundary(meta, body)
    s4 = score_checkpoint(meta, body)
    s5 = score_specificity(meta, body)
    s6 = score_resources(meta, skill_dir)
    s7 = score_compatibility(meta)
    structural_total = s1 + s2 + s3 + s4 + s5 + s6 + s7

    # 业务维度（50分）
    s8 = score_business_relevance(meta)
    s9 = score_uniqueness(meta)
    s10 = score_executability(meta)
    s11 = score_impact(meta)
    s12 = score_maturity(meta)
    business_total = s8 + s9 + s10 + s11 + s12

    total = structural_total + business_total

    # Tier 划分
    if name in DEFAULT_TIER1_SKILLS:
        tier = 1
    elif total >= 80:
        tier = 1
    elif total >= 65:
        tier = 2
    elif total >= 50:
        tier = 3
    else:
        tier = 0  # 不收录

    return {
        "name": name,
        "description": meta.get("description", "")[:120],
        "scores": {
            "frontmatter": s1,
            "workflow": s2,
            "boundary": s3,
            "checkpoint": s4,
            "specificity": s5,
            "resources": s6,
            "compatibility": s7,
            "business_relevance": s8,
            "uniqueness": s9,
            "executability": s10,
            "impact": s11,
            "maturity": s12,
        },
        "structural_total": structural_total,
        "business_total": business_total,
        "total": total,
        "tier": tier,
    }


# ── 批量评估 ────────────────────────────────────────────────────────────

def find_all_skill_mds(root_dir: Path) -> list[Path]:
    """递归查找所有 SKILL.md 文件。"""
    return sorted(root_dir.rglob("SKILL.md"))


def batch_evaluate(skill_mds: list[Path]) -> list[dict[str, Any]]:
    """批量评估多个 SKILL.md，返回评分报告列表。"""
    results = []
    seen_names: set[str] = set()
    for md_path in skill_mds:
        meta = parse_skill_md(md_path)
        if not meta or not meta.get("name"):
            continue
        name = meta["name"]
        if name in seen_names:
            continue
        seen_names.add(name)

        body = md_path.read_text(encoding="utf-8", errors="ignore")
        body = body[body.find("---", 3) + 3:] if body.startswith("---") else body
        score = calculate_score(meta, body, md_path.parent)
        score["file_path"] = str(md_path)
        results.append(score)
    return results


def build_tiers(results: list[dict[str, Any]]) -> dict[str, dict]:
    """从评分结果构建分级字典。"""
    tiers: dict[str, dict] = {"tier1": {}, "tier2": {}, "tier3": {}}
    for r in results:
        if r["tier"] == 1:
            tiers["tier1"][r["name"]] = {
                "description": r["description"],
                "score": r["total"],
                "file_path": r.get("file_path", ""),
            }
        elif r["tier"] == 2:
            tiers["tier2"][r["name"]] = {
                "description": r["description"],
                "score": r["total"],
                "file_path": r.get("file_path", ""),
            }
        elif r["tier"] == 3:
            tiers["tier3"][r["name"]] = {
                "description": r["description"],
                "score": r["total"],
                "file_path": r.get("file_path", ""),
            }
    return tiers


# ── 命令行 ────────────────────────────────────────────────────────────────

def print_report(result: dict[str, Any]):
    """打印单个评分报告。"""
    s = result["scores"]
    print(f"\n{'='*50}")
    print(f"  {result['name']}  |  总分: {result['total']}/100  |  Tier {result['tier']}")
    print(f"  {result['description'][:80]}")
    print(f"{'='*50}")
    print(f"  结构维度 ({result['structural_total']}/50):")
    print(f"    Frontmatter: {s['frontmatter']}/5  |  工作流: {s['workflow']}/10")
    print(f"    边界: {s['boundary']}/8  |  检查点: {s['checkpoint']}/5")
    print(f"    指令: {s['specificity']}/12  |  资源: {s['resources']}/5")
    print(f"    兼容性: {s['compatibility']}/5")
    print(f"  业务维度 ({result['business_total']}/50):")
    print(f"    相关性: {s['business_relevance']}/12  |  独特性: {s['uniqueness']}/12")
    print(f"    可执行: {s['executability']}/10  |  影响力: {s['impact']}/10")
    print(f"    成熟度: {s['maturity']}/6")


def main():
    parser = argparse.ArgumentParser(
        description="Skills 11维评价工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("path", nargs="?", help="SKILL.md 文件路径或目录路径")
    parser.add_argument("--batch", type=str, help="批量评估目录下所有 SKILL.md")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    parser.add_argument("--output", type=str, help="输出 JSON 到文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    if args.batch:
        root = Path(args.batch)
        if not root.is_dir():
            print(f"错误: 目录不存在: {root}", file=sys.stderr)
            sys.exit(1)
        skill_mds = find_all_skill_mds(root)
        if not skill_mds:
            print(f"未找到 SKILL.md", file=sys.stderr)
            sys.exit(1)
        results = batch_evaluate(skill_mds)

        tiers = build_tiers(results)
        if args.json or args.output:
            if args.output:
                out_path = Path(args.output)
                out_path.write_text(json.dumps(tiers, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"JSON 报告已写入: {out_path}")
            else:
                print(json.dumps(tiers, ensure_ascii=False, indent=2))
        else:
            print(f"\n评价完成: {len(results)} 个 skills")
            print(f"  Tier 1 (核心): {len(tiers['tier1'])}")
            print(f"  Tier 2 (扩展): {len(tiers['tier2'])}")
            print(f"  Tier 3 (长尾): {len(tiers['tier3'])}")
            for r in results:
                print_report(r)

        # 统计
        t1 = len(tiers["tier1"])
        t2 = len(tiers["tier2"])
        t3 = len(tiers["tier3"])
        print(f"\n{'='*50}")
        print(f"  总计: {len(results)}  |  Tier 1: {t1}  |  Tier 2: {t2}  |  Tier 3: {t3}")
        print(f"{'='*50}")

    elif args.path:
        path = Path(args.path)
        if not path.exists():
            print(f"错误: 路径不存在: {path}", file=sys.stderr)
            sys.exit(1)
        if path.is_dir():
            skill_mds = find_all_skill_mds(path)
            results = batch_evaluate(skill_mds)
        else:
            meta = parse_skill_md(path)
            if not meta:
                print("无法解析 SKILL.md", file=sys.stderr)
                sys.exit(1)
            body = path.read_text(encoding="utf-8", errors="ignore")
            body = body[body.find("---", 3) + 3:] if body.startswith("---") else body
            result = calculate_score(meta, body, path.parent)
            results = [result]

        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for r in results:
                print_report(r)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
