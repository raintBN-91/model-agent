"""MemoryEngine — 工作流执行记忆的提取、检索、注入。

从每次工作流执行中自动提取关键事实（模型名、平台、算子、方案），
存入 ExperienceStore，下次同类任务时自动注入到 LLM prompt 中。
"""

from __future__ import annotations

import json
import re
from typing import Any

from engine.experience.store import ExperienceStore
from engine.workflow.types import ExecutionContext, StepStatus, WorkflowPlan


# 常见昇腾模型名模式
_MODEL_PATTERNS = [
    r"Qwen3?\.?\d*-?\w*",
    r"Llama\d*\.?\d*-?\w*",
    r"DeepSeek-?\w*",
    r"Boltz\d*-?\w*",
    r"Gemma\d*-?\w*",
    r"Mistral-?\w*",
    r"Mixtral-?\w*",
    r"ChatGLM-?\w*",
    r"StableDiffusion-?\w*",
    r"Whisper-?\w*",
]

# 昇腾平台名
_PLATFORM_PATTERNS = [r"\bA\d+\b", r"\bAscend\w*\b", r"\bNPU\b"]


def _extract_model(intent: str, goal: str = "") -> str:
    combined = f"{intent} {goal}"
    for p in _MODEL_PATTERNS:
        m = re.search(p, combined, re.IGNORECASE)
        if m:
            return m.group(0)
    return ""


def _extract_platform(intent: str, goal: str = "") -> str:
    combined = f"{intent} {goal}"
    for p in _PLATFORM_PATTERNS:
        m = re.search(p, combined)
        if m:
            return m.group(0)
    return ""


def _extract_key_facts(plan: WorkflowPlan) -> dict[str, Any]:
    """从工作流步骤中提取关键事实。"""
    facts: dict[str, Any] = {}
    for s in plan.steps:
        if s.skill_name:
            facts.setdefault("skills", []).append(s.skill_name)
        if "mcp_server" in (s.params or {}):
            facts.setdefault("mcp_servers", []).append(s.params["mcp_server"])
        if s.type and s.type.value == "mcp_tool":
            facts.setdefault("mcp_tools", []).append(s.skill_name)
    for key in list(facts.keys()):
        if isinstance(facts[key], list):
            facts[key] = list(set(facts[key]))
    return facts


class MemoryEngine:
    """记忆引擎 — 提取、检索、压缩、prompt 注入。"""

    def __init__(self, store: ExperienceStore):
        self._store = store

    def extract(self, plan: WorkflowPlan, ctx: ExecutionContext) -> dict | None:
        """从工作流执行结果中提取一条记忆。

        Returns:
            记忆 dict，如果没有任何步骤执行则返回 None。
        """
        steps = plan.steps
        if not steps:
            return None

        step_summaries = []
        error_count = 0
        for s in steps:
            dur = None
            if s.started_at and s.completed_at:
                dur = round(s.completed_at - s.started_at, 1)
            if s.status == StepStatus.FAILED:
                error_count += 1
            step_summaries.append({
                "type": s.type.value if s.type else "unknown",
                "skill": s.skill_name or "",
                "status": s.status.value if s.status else "unknown",
                "duration": dur,
                "error": s.error[:120] if s.error else None,
            })

        intent = plan.intent or ""
        goal = plan.goal or ""

        memory = {
            "intent": intent[:200],
            "goal": goal[:200],
            "model": _extract_model(intent, goal),
            "platform": _extract_platform(intent, goal),
            "steps": step_summaries,
            "result": "success" if error_count == 0 else "partial",
            "key_facts": _extract_key_facts(plan),
            "duration": round(
                sum((s.get("duration") or 0) for s in step_summaries), 1
            ),
            "error_count": error_count,
            "step_count": len(steps),
            "source": "workflow",
        }
        return memory

    def retrieve(self, user_input: str, top_k: int = 5) -> list[dict]:
        """检索与用户输入最相关的历史记忆。

        策略:
        1. 从输入中提取模型名/平台名
        2. 精确匹配模型名
        3. 关键词匹配兜底
        """
        memories = self._store.get_memories(limit=500)
        if not memories:
            return []

        model = _extract_model(user_input)
        platform = _extract_platform(user_input)
        query_lower = user_input.lower()
        query_words = set(query_lower.split())

        scored: list[tuple[float, dict]] = []
        for m in memories:
            score = 0.0

            # 模型名精确匹配
            if model and model.lower() in m.get("intent", "").lower():
                score += 10.0
            if model and model.lower() in m.get("model", "").lower():
                score += 20.0

            # 平台匹配
            if platform and platform in m.get("platform", ""):
                score += 10.0

            # 关键词匹配
            mem_text = json.dumps(m, ensure_ascii=False).lower()
            keyword_hits = sum(1 for w in query_words if len(w) > 2 and w in mem_text)
            score += keyword_hits * 2.0

            # 最近优先（时间衰减）
            if score > 0:
                scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def format_prompt(self, user_input: str, max_entries: int = 3) -> str:
        """将相关记忆格式化为 LLM prompt 的 Experience 章节。

        返回格式:
        ## 历史经验参考

        以下是与当前任务相关的历史执行经验：
        - [2026-06-10] 适配 Qwen3-8B → A2：使用 ascendc-op-develop 开发缺失算子，成功
        - [2026-06-09] Profiling Qwen3-8B → A2：使用 profiling_analyze，发现瓶颈在 Attention
        """
        memories = self.retrieve(user_input, top_k=max_entries)
        if not memories:
            return "暂无历史经验。"

        lines = []
        for m in memories:
            ts = (m.get("timestamp") or "?")[:10]
            intent = (m.get("intent") or "?")[:80]
            result = m.get("result", "?")
            model = m.get("model", "")
            platform = m.get("platform", "")
            key_info = f"{model} → {platform}" if model and platform else intent
            lines.append(f"- [{ts}] {key_info}：{intent} → {result}")

        return "\n".join(lines)

    def compress(self, memories: list[dict], max_chars: int = 500) -> str:
        """将多条记忆压缩为紧凑的单行摘要。"""
        parts: list[str] = []
        for m in memories:
            ts = (m.get("timestamp") or "?")[:10]
            model = m.get("model", "")
            platform = m.get("platform", "")
            result = m.get("result", "?")
            key = f"{model}/{platform}" if model and platform else m.get("intent", "?")[:40]
            parts.append(f"[{ts}] {key} → {result}")

        result = " | ".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars] + "…"
        return result

    def trim_to_capacity(self, capacity: int = 500) -> int:
        """裁减记忆到容量限制。"""
        return self._store.trim_memories(capacity)
