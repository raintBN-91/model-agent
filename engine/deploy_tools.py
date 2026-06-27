"""deploy-agent 工具模块。"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def deploy_model_deployment(model_name: str, model_path: str = "", device: str = "npu") -> str:
    """在昇腾 NPU 上部署模型，进行服务化推理。

    Args:
        model_name: 要部署的模型名称。
        model_path: 模型文件路径，默认为空（根据模型名称自动查找）。
        device: 运行设备，默认 "npu"。
    """
    return (
        f"[deploy-agent] 模型部署功能待实现。"
        f" model_name={model_name!r}, model_path={model_path!r}, device={device!r}"
    )


@tool
def deploy_esm2_demo(sequence: str = "", output_format: str = "embeddings") -> str:
    """运行 ESM2 蛋白质语言模型演示。

    Args:
        sequence: 蛋白质氨基酸序列，默认提供示例序列。
        output_format: 输出格式，可选 "embeddings" 或 "predictions"，默认 "embeddings"。
    """
    return (
        f"[deploy-agent] ESM2 演示功能待实现。"
        f" sequence={sequence!r}, output_format={output_format!r}"
    )


def get_tools() -> list:
    return [deploy_model_deployment, deploy_esm2_demo]
