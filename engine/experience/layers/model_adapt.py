"""Layer A: 模型适配经验积累。

每次适配执行后自动提取模型-平台-框架-算子的映射关系，
下次同类任务时 DynamicPlanner 自动注入这些参数到 prompt 中。
"""

from __future__ import annotations

import re
from typing import Any

from engine.experience.store import ExperienceStore
from engine.workflow.types import ExecutionContext, WorkflowPlan


# 常见适配框架关键词
_FRAMEWORK_KEYWORDS = {
    "vllm": "vLLM-Ascend",
    "vllm-ascend": "vLLM-Ascend",
    "sglang": "SGLang",
    "mindie": "MindIE",
    "atb": "ATB",
    "pytorch": "PyTorch",
    "mindspore": "MindSpore",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
}

# 常见量化方式
_QUANT_KEYWORDS = {
    "fp16": "FP16",
    "fp32": "FP32",
    "int8": "INT8",
    "int4": "INT4",
    "bf16": "BF16",
    "fp8": "FP8",
    "w8a16": "W8A16",
    "awq": "AWQ",
    "gptq": "GPTQ",
}


def _detect_framework(intent: str, steps: list) -> str:
    combined = f"{intent} {' '.join(s.get('skill', '') for s in steps)}".lower()
    for kw, name in _FRAMEWORK_KEYWORDS.items():
        if kw in combined:
            return name
    return ""


def _detect_quantization(intent: str) -> str:
    intent_lower = intent.lower()
    for kw, name in _QUANT_KEYWORDS.items():
        if kw in intent_lower:
            return name
    return ""


def _extract_missing_ops(steps: list) -> list[str]:
    """从步骤结果中提取缺失算子名称。"""
    ops = set()
    for s in steps:
        result = s.get("result") or ""
        if isinstance(result, str):
            found = re.findall(r"(?:缺失算子|missing op|op)[：:\s]*(\w+)", result, re.IGNORECASE)
            ops.update(found)
    return sorted(ops)


class ModelAdaptLayer:
    """Layer A: 模型适配经验积累。

    每次适配执行后自动提取模型-平台-框架的映射，保存到 ExperienceStore。
    下次同类适配时自动提示已知参数。
    """

    def __init__(self, store: ExperienceStore):
        self._store = store

    def extract_from_workflow(
        self, plan: WorkflowPlan, ctx: ExecutionContext
    ) -> dict | None:
        """从工作流执行结果提取适配参数。"""
        intent = plan.intent or ""
        if not any(kw in intent.lower() for kw in ["适配", "adapt", "移植", "算子", "op"]):
            return None

        steps_data = []
        for s in plan.steps:
            result = ctx.step_results.get(s.id, "")
            steps_data.append({
                "type": s.type.value if s.type else "",
                "skill": s.skill_name or "",
                "status": s.status.value if s.status else "",
                "result": str(result)[:500] if result else "",
            })

        entry = {
            "model": plan.goal[:100] if plan.goal else intent[:100],
            "platform": "",
            "last_adapted": __import__("datetime").datetime.now().isoformat(),
            "framework": _detect_framework(intent, steps_data),
            "quantization": _detect_quantization(intent),
            "missing_ops": _extract_missing_ops(steps_data),
            "step_count": len(plan.steps),
            "success": all(
                s.status.name == "COMPLETED" for s in plan.steps
            ),
        }
        return entry

    def suggest_params(self, model: str, platform: str) -> dict | None:
        """为已知模型 + 平台组合建议适配参数。"""
        data = self._store.load_layer_data("model_adapt")
        if not data:
            return None

        entries = data.get("entries", [data])  # 兼容单条和数组
        if isinstance(entries, dict):
            entries = [entries]

        model_lower = model.lower()
        platform_lower = platform.lower()
        for e in entries:
            e_model = (e.get("model") or "").lower()
            e_platform = (e.get("platform") or "").lower()
            if model_lower in e_model and platform_lower in e_platform:
                return {
                    "framework": e.get("framework", ""),
                    "quantization": e.get("quantization", ""),
                    "missing_ops": e.get("missing_ops", []),
                    "known_good": e.get("success", False),
                }
        return None
