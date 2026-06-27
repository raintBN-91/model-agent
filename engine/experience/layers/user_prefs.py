"""Layer C: 用户偏好自适应。

从历史执行中隐式推断用户的偏好（精度、文档详细程度、风险偏好等），
在规划时自动调整 prompt 方向。
"""

from __future__ import annotations

from typing import Any

from engine.experience.store import ExperienceStore


class UserPrefsLayer:
    """用户偏好自适应层。"""

    def __init__(self, store: ExperienceStore):
        self._store = store

    def infer_from_history(self, memories: list[dict]) -> dict:
        """从历史执行中推断用户偏好。"""
        prefs: dict[str, Any] = {
            "precision": "",       # fp32 / fp16 / int8
            "doc_detail": "normal",  # brief / normal / detailed
            "optim_target": "",    # throughput / latency
            "risk_tolerance": "normal",  # conservative / normal / aggressive
        }

        for m in memories:
            intent = (m.get("intent") or "").lower()
            # 检测精度偏好
            if "int8" in intent or "量化" in intent:
                prefs["precision"] = "INT8"
            elif "fp16" in intent or "混合精度" in intent:
                prefs["precision"] = "FP16"
            # 检测优化目标
            if "吞吐" in intent or "throughput" in intent:
                prefs["optim_target"] = "throughput"
            elif "延迟" in intent or "latency" in intent:
                prefs["optim_target"] = "latency"
            # 检测文档偏好
            if "详细文档" in intent or "完整文档" in intent:
                prefs["doc_detail"] = "detailed"

        return prefs

    def format_prefs(self, prefs: dict) -> str:
        """将偏好格式化为 prompt。"""
        parts = []
        if prefs.get("precision"):
            parts.append(f"精度偏好: {prefs['precision']}")
        if prefs.get("optim_target"):
            parts.append(f"优化目标: {prefs['optim_target']}")
        if prefs.get("doc_detail") == "detailed":
            parts.append("需要详细文档输出")
        return "；".join(parts) if parts else ""
