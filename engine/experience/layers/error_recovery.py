from __future__ import annotations
"""Layer B: 错误恢复模式学习。

记录失败模式 + 恢复策略，下次遇到相同错误时自动建议恢复方案。
"""


from typing import Any

from engine.experience.store import ExperienceStore
from engine.workflow.types import WorkflowStep


class ErrorRecoveryLayer:
    """错误恢复模式学习层。"""

    def __init__(self, store: ExperienceStore):
        self._store = store

    def record_failure(
        self, step: WorkflowStep, error: str, recovery: str, success: bool
    ) -> None:
        """记录一次失败和恢复策略。"""
        self._store.append_layer_array("error_recovery", {
            "step_type": step.type.value if step.type else "",
            "skill_name": step.skill_name or "",
            "error": error[:200],
            "recovery": recovery[:200],
            "success": success,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        })

    def suggest_recovery(
        self, step_type: str, skill_name: str, error: str
    ) -> str | None:
        """根据历史匹配建议恢复策略。"""
        data = self._store.load_layer_data("error_recovery")
        entries = data.get("entries", [])
        if not entries:
            return None

        error_lower = error.lower()[:100]
        for e in reversed(entries):
            if not e.get("success"):
                continue
            err = (e.get("error") or "").lower()[:100]
            # 错误文本相似度匹配
            common = set(error_lower.split()) & set(err.split())
            if len(common) >= 3 and e.get("skill_name") == skill_name:
                return e.get("recovery", "")
        return None
