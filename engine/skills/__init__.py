"""Skills 三级注册系统 — 分层管理 model-agent 和内置 skills。

SkillRegistry 管理三个层级的 skills:
  - Tier 1 (核心): 始终注入 planner 的 system prompt（v0.3.0 内置 + model-agent ≥80 分）
  - Tier 2 (扩展): 名称列表进入 prompt，LLM 引用后自动展开
  - Tier 3 (长尾): 不进 prompt，通过 /skill-search 发现
"""
from __future__ import annotations

from engine.skills.registry import SkillRegistry

__all__ = ["SkillRegistry"]
