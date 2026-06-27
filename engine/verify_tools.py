"""verify-agent 工具模块（骨架）。"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def verify_run_benchmark(model_name: str, npu_ids: str = "0,1,2,3") -> str:
    """在昇腾 NPU 上对指定模型运行 vLLM 基准测试。（骨架，待实现）

    Args:
        model_name: 模型名称，如 "Qwen3-8B"。
        npu_ids: 使用的 NPU 编号，逗号分隔，默认 "0,1,2,3"。
    """
    return f"[verify-agent] 暂未接入，model={model_name!r}, npu_ids={npu_ids!r}"


@tool
def verify_check_npu_status() -> str:
    """检查当前昇腾 NPU 设备状态（健康度、占用情况）。（骨架，待实现）"""
    return "[verify-agent] 暂未接入，NPU 状态检查待实现"


def get_tools() -> list:
    return [verify_run_benchmark, verify_check_npu_status]
