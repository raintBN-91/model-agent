"""search-agent 工具模块。"""

from __future__ import annotations

import os, sys
from pathlib import Path

from langchain_core.tools import tool

from .search_result_parser import parse_and_format_search_results

# search-agent/src 路径查找（按优先级）：
# 1. 环境变量 SEARCH_AGENT_SRC
# 2. <project_root>/../search-agent/src （同目录兄弟仓库）
# 3. <project_root>/search-agent/src （仓库内子目录）
_SEARCH_SRC = None
_env_src = os.environ.get("SEARCH_AGENT_SRC")
_project_root = Path(__file__).resolve().parents[1]
_candidates = [
    _env_src,
    _project_root.parent / "search-agent" / "src",
    _project_root / "search-agent" / "src",
]
for _c in _candidates:
    if _c and Path(_c).is_dir():
        _SEARCH_SRC = Path(_c)
        break
if _SEARCH_SRC and str(_SEARCH_SRC) not in sys.path:
    sys.path.insert(0, str(_SEARCH_SRC))

try:
    from run_dify_workflow_direct import run_dify
except Exception as _import_err:
    import logging

    logging.getLogger(__name__).warning(
        "Failed to import run_dify_workflow_direct from %s: %s",
        _SEARCH_SRC,
        _import_err,
    )
    run_dify = None  # type: ignore[assignment]


@tool
def search_ascend_models(input: str = "") -> str:
    """从 GitCode 昇腾模型库搜索已适配昇腾 NPU 的模型。

    支持两种输入方式：
    1. 直接输入模型关键字：如 "GLM 4.5"、"Qwen3-14B"
    2. 输入自然语言指令：系统会自动提取模型名、适配框架、硬件等关键字进行搜索。

    Args:
        input: 搜索关键词或自然语言指令，留空返回全量列表。
    """
    if run_dify is None:
        return (
            f"[search-agent] 无法加载 search-agent 依赖。"
            f"搜索路径: {_SEARCH_SRC or '未找到（已尝试环境变量 SEARCH_AGENT_SRC、项目同级目录、项目内子目录）'}，"
            f"请检查该目录下是否存在 run_dify_workflow_direct.py 及其依赖。\n"
            f"可通过 export SEARCH_AGENT_SRC=/path/to/search-agent/src 设置正确路径。"
        )
    try:
        result = run_dify(input)
        raw_output = str(result.get("answer", result)) if isinstance(result, dict) else str(result)
        return parse_and_format_search_results(raw_output)
    except Exception as e:
        return f"[search-agent] 调用失败：{e}"


@tool
def search_agent_example_glm5() -> str:
    """从 GitCode GLM5 示例库搜索相关示例（固定输入演示）。"""
    if run_dify is None:
        return "[search-agent] 环境未安装 search-agent 依赖。"
    try:
        return str(run_dify("请帮我查询GLM5 适配vllm-ascend框架 A2硬件的部署文档"))
    except Exception as e:
        return f"[search-agent] example_glm5 调用失败：{e}"


def get_tools() -> list:
    return [search_ascend_models, search_agent_example_glm5]
