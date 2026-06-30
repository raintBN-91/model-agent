"""optimizer-agent 工具模块。"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def optimizer_torch_npu_optimize(model_files: str, op_types: str = "rmsnorm,swiglu,rope") -> str:
    """对 PyTorch 模型文件进行 torch_npu 融合算子替换，提升昇腾 NPU 推理性能。

    Args:
        model_files: 需要优化的模型 Python 文件路径，多个用逗号分隔。
        op_types: 要替换的算子类型，逗号分隔，默认 "rmsnorm,swiglu,rope"。
    """
    return (
        f"[optimizer-agent] torch_npu 推理优化待实现。"
        f" model_files={model_files!r}, op_types={op_types!r}"
    )


@tool
def optimizer_npu_profiling(
    script_path: str, level: str = "L1", output_dir: str = "./Profiling_output"
) -> str:
    """在昇腾 NPU 上对推理/训练脚本进行性能 profiling，输出算子耗时分析。

    Args:
        script_path: 待 profiling 的 Python 脚本路径。
        level: 采集级别，"L0" / "L1" / "L2"，默认 "L1"。
        output_dir: profiling 结果输出目录，默认 "./Profiling_output"。
    """
    return (
        f"[optimizer-agent] NPU profiling 待实现。"
        f" script={script_path!r}, level={level!r}, output={output_dir!r}"
    )


@tool
def optimizer_triton_op_optimize(
    op_file: str,
    perf_test_file: str,
    target_speedup: float = 2.0,
    output_dir: str = "./triton_opt_output",
) -> str:
    """对昇腾 NPU 上的 Vector 类 Triton 算子进行深度性能优化。

    Args:
        op_file: Triton 算子源文件路径（如 rmsnorm.py）。
        perf_test_file: 性能测试文件路径（如 test_rmsnorm_perf.py）。
        target_speedup: 目标加速比，默认 2.0。
        output_dir: 优化结果与 profiling 数据输出目录。
    """
    return (
        f"[optimizer-agent] Triton 算子优化待实现。"
        f" op={op_file!r}, target={target_speedup}x, output={output_dir!r}"
    )


def get_tools() -> list:
    return [
        optimizer_torch_npu_optimize,
        optimizer_npu_profiling,
        optimizer_triton_op_optimize,
    ]
