"""quantify-agent 工具模块（骨架）。"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def quantify_run_msmodelslim(model_path: str, quant_type: str = "W8A8") -> str:
    """使用 msmodelslim 对模型进行量化。（骨架，待实现）

    Args:
        model_path: 模型本地路径或 ModelScope/HuggingFace 模型 ID。
        quant_type: 量化类型，如 "W8A8"、"W4A8"、"W4A4"，默认 "W8A8"。
    """
    return f"[quantify-agent] 暂未接入，model_path={model_path!r}, quant_type={quant_type!r}"


@tool
def quantify_sensitivity_analysis(model_path: str, dataset: str = "gsm8k") -> str:
    """对模型进行量化敏感度分析，识别精度下降的层。（骨架，待实现）

    Args:
        model_path: 模型本地路径。
        dataset: 评估数据集，默认 "gsm8k"。
    """
    return f"[quantify-agent] 暂未接入，model_path={model_path!r}, dataset={dataset!r}"


def get_tools() -> list:
    return [quantify_run_msmodelslim, quantify_sensitivity_analysis]
