from __future__ import annotations
"""SkillRegistry — 三级技能注册表。

职责:
  1. 加载 agents/model-agent/tiers.json 分级配置
  2. 发现 skills/ 目录下所有 v0.3.0 内置 SKILL.md
  3. 为 DynamicPlanner 提供层级化的 prompt 注入
  4. Tier 2 技能引用膨胀、Tier 3 技能搜索

分级:
  Tier 1 (核心) — 完整描述注入 planner prompt
  Tier 2 (扩展) — 仅名称列表进入 prompt，引用时膨胀
  Tier 3 (长尾) — 不进 prompt，/skill-search 检索
"""


import json
import re
from pathlib import Path
from typing import Any


# ── 内置 Tier 1 技能（v0.3.0）─────────────────────────────────────────

DEFAULT_TIER1_SKILLS: set[str] = {
    "adapt-agent", "verify-agent", "optimizer-agent", "quantify-agent",
    "ai4s-main", "ai4s-basic", "ai4s-perf-tuning", "ai4s-precision-alignment",
    "ai4s-profiling", "ascend-optimization", "ascend-affinity-operator",
    "ascend-history-to-skill", "gitcode-publish",
}


def _parse_frontmatter(text: str) -> dict[str, str] | None:
    """从 SKILL.md 提取 name 和 description。"""
    if not text.startswith("---"):
        return None
    end = text.find("---", 3)
    if end == -1:
        return None
    front = text[3:end]
    name: str | None = None
    desc_lines: list[str] = []
    in_desc = False
    for line in front.splitlines():
        stripped = line.strip()
        if stripped.startswith("name:"):
            name = stripped.split(":", 1)[1].strip().strip('"').strip("'")
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
    if name:
        return {"name": name, "description": " ".join(desc_lines) if desc_lines else name}
    return None


def _scan_skills_dir(skills_dir: Path) -> dict[str, str]:
    """扫描目录下所有 SKILL.md，返回 {name: description}。"""
    registry: dict[str, str] = {}
    if not skills_dir.is_dir():
        return registry
    for skill_md in sorted(skills_dir.rglob("SKILL.md")):
        text = skill_md.read_text(encoding="utf-8", errors="ignore")
        parsed = _parse_frontmatter(text)
        if parsed and parsed["name"] and parsed["name"] not in registry:
            registry[parsed["name"]] = parsed["description"]
    return registry


# ── 项目路径 ──────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SKILLS_DIR = _PROJECT_ROOT / "skills"
_TIERS_JSON = _PROJECT_ROOT / "skills" / "tiers.json"


# ── SkillRegistry ─────────────────────────────────────────────────────

class SkillRegistry:
    """三级技能注册表 — 为 DynamicPlanner 提供 tier 感知的技能列表。"""

    def __init__(self) -> None:
        # name → description（不跨 tier 去重，同名以高 tier 为准）
        self._tier1: dict[str, str] = {}
        self._tier2: dict[str, str] = {}
        self._tier3: dict[str, str] = {}
        # 聚合视图 — 所有层级的 name → (description, tier)
        self._all: dict[str, tuple[str, int]] = {}
        self._loaded = False

    # ── 加载入口 ──────────────────────────────────────────────────────

    def load(self) -> None:
        """加载所有技能来源，构建三级索引。

        加载顺序:
          1. tiers.json（model-agent 分级结果）
          2. skills/（v0.3.0 内置，强制 Tier 1）
          3. agents/model-agent/skills/（扫描 fallback，若 tiers.json 不存在）
        """
        if self._loaded:
            return

        # 1. 尝试加载 tiers.json
        tiers_data = self._load_tiers_json()
        if tiers_data:
            self._apply_tiers(tiers_data)
        else:
            # 2. fallback: 扫描 model-agent skills 目录
            self._scan_model_agent_fallback()

        # 3. 强制注入内置 Tier 1（v0.3.0 skills）
        self._load_builtin_tier1()

        self._loaded = True

    # ── 数据加载方法 ──────────────────────────────────────────────────

    def _load_tiers_json(self) -> dict | None:
        """读取 agents/model-agent/tiers.json。"""
        if not _TIERS_JSON.exists():
            return None
        try:
            return json.loads(_TIERS_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _apply_tiers(self, data: dict) -> None:
        """将 tiers.json 数据写入对应层级字典。"""
        for tier_key, tier_num in [("tier1", 1), ("tier2", 2), ("tier3", 3)]:
            skills = data.get(tier_key, {})
            target = getattr(self, f"_tier{tier_num}")
            for name, info in skills.items():
                desc = info.get("description", "") if isinstance(info, dict) else ""
                if name not in self._all:  # 较高 tier 优先
                    target[name] = desc
                    self._all[name] = (desc, tier_num)

    def _scan_model_agent_fallback(self) -> None:
        """没有 tiers.json 时直接扫描目录，全部归入 Tier 2。"""
        if not _SKILLS_DIR.is_dir():
            return
        registry = _scan_skills_dir(_SKILLS_DIR)
        for name, desc in registry.items():
            if name not in self._all:
                self._tier2[name] = desc
                self._all[name] = (desc, 2)

    def _load_builtin_tier1(self) -> None:
        """扫描 skills/ 目录，内置技能全部注入 Tier 1。"""
        registry = _scan_skills_dir(_SKILLS_DIR)
        for name, desc in registry.items():
            if name not in self._all:
                self._tier1[name] = desc
                self._all[name] = (desc, 1)
            elif self._all[name][1] > 1:
                # 同名但当前在低 tier，提升到 Tier 1
                self._tier1[name] = desc
                # 从低 tier 移除
                for t in (2, 3):
                    getattr(self, f"_tier{t}").pop(name, None)
                self._all[name] = (desc, 1)

    # ── Prompt 生成 ──────────────────────────────────────────────────

    def format_prompt(self, tier2_context: dict[str, str] | None = None) -> str:
        """生成 planner 的 skills_section。

        Args:
            tier2_context: LLM 在 planning 时引用到的 Tier 2 skill 映射
                          {skill_name: expanded_prompt_text}，这些会以完整
                          形式附加在 Tier 1 列表之后。

        Returns:
            可直接注入 system prompt 的文本。
        """
        self.load()
        parts: list[str] = []

        # Tier 1 — 完整列表
        if self._tier1:
            parts.append("### Tier 1（核心技能 — 可直接选用）")
            for name, desc in sorted(self._tier1.items()):
                short = desc[:150].replace("\n", " ")
                parts.append(f"- `{name}`: {short}")
            parts.append("")

        # Tier 2 已展开 — 附加在 Tier 1 之后
        expanded = tier2_context or {}
        if expanded:
            parts.append("### Tier 2（已展开的扩展技能）")
            for name, text in sorted(expanded.items()):
                short = text[:150].replace("\n", " ")
                parts.append(f"- `{name}`: {short}")
            parts.append("")

        # Tier 2 未展开 — 仅名称列表
        uninflated = {k: v for k, v in self._tier2.items() if k not in expanded}
        if uninflated:
            parts.append("### Tier 2（扩展技能 — 按需引用）")
            parts.append("以下技能仅列出名称，如需使用可在 plan 中引用 skill_name，系统将自动展开：")
            names = ", ".join(sorted(uninflated.keys()))
            parts.append(names)
            parts.append("")

        if not parts:
            return "暂无可用 Skill。"

        return "\n".join(parts)

    def format_tier2_names(self) -> str:
        """仅返回 Tier 2 技能名称列表（用于 planner 简洁模式）。"""
        self.load()
        if not self._tier2:
            return ""
        return "可引用的扩展技能: " + ", ".join(sorted(self._tier2.keys()))

    # ── 技能检索 ──────────────────────────────────────────────────────

    def expand(self, name: str) -> str | None:
        """展开 Tier 2 技能为完整描述文本。

        当 DynamicPlanner 生成的 plan 中引用了某个 Tier 2 skill 时，
        executor 可调用此方法获取完整描述传递给 LLM。
        """
        self.load()
        desc = self._tier2.get(name)
        if desc:
            return f"### {name}\n\n{desc}"
        # 也允许展开 Tier 1 / Tier 3（容错）
        desc = self._tier1.get(name)
        if desc:
            return f"### {name}\n\n{desc}"
        return None

    def expand_multi(self, names: list[str]) -> dict[str, str]:
        """批量展开多个技能。"""
        return {n: text for n in names if (text := self.expand(n))}

    def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """在所有层级中搜索技能（忽略大小写）。

        Args:
            query: 搜索关键词
            max_results: 最大返回数

        Returns:
            [{name, description, tier, score_match}, ...]
        """
        self.load()
        q = query.lower()
        results: list[dict[str, Any]] = []

        for name, (desc, tier) in self._all.items():
            score = 0
            if q in name.lower():
                score += 10
            if q in desc.lower():
                score += 5
            # 按词匹配
            for word in q.split():
                if len(word) > 2 and word in name.lower():
                    score += 3
                if len(word) > 2 and word in desc.lower():
                    score += 1
            if score > 0:
                results.append({
                    "name": name,
                    "description": desc[:200],
                    "tier": tier,
                    "score": score,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]

    def get(self, name: str) -> str | None:
        """按名称获取技能描述（跨所有层级）。"""
        self.load()
        info = self._all.get(name)
        return info[0] if info else None

    def get_tier(self, name: str) -> int:
        """查询技能所属层级（0 = 不存在）。"""
        self.load()
        info = self._all.get(name)
        return info[1] if info else 0

    @property
    def count(self) -> dict[str, int]:
        """返回各层级数量统计。"""
        self.load()
        return {
            "tier1": len(self._tier1),
            "tier2": len(self._tier2),
            "tier3": len(self._tier3),
            "total": len(self._all),
        }

    # ── 工具方法 ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """导出完整注册表（用于调试 / 序列化）。"""
        self.load()
        return {
            "tier1": {n: d for n, d in sorted(self._tier1.items())},
            "tier2": {n: d for n, d in sorted(self._tier2.items())},
            "tier3": {n: d for n, d in sorted(self._tier3.items())},
        }


# ── 全局单例 ──────────────────────────────────────────────────────────

_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    """获取全局 SkillRegistry 单例。"""
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
        _registry.load()
    return _registry
