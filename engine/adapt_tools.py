"""adapt-agent 工具模块（骨架）。"""

from __future__ import annotations

import json

from langchain_core.tools import tool


@tool
def adapt_review_repo(repo_url_or_path: str) -> str:
    """对 GPU 代码仓库进行昇腾 NPU 适配审查，生成 Markdown 报告。（骨架，待实现）

    Args:
        repo_url_or_path: GitHub 仓库 URL 或本地代码路径。
    """
    return json.dumps({
        "status": "not_implemented",
        "message": f"adapt_review_repo 暂未接入（repo={repo_url_or_path!r}）",
        "tool": "adapt_review_repo",
    }, ensure_ascii=False)


@tool
def adapt_generate_compat_layer(repo_path: str) -> str:
    """为指定仓库生成 NPU 兼容适配层代码（npu_compat.py 等）。（骨架，待实现）

    Args:
        repo_path: 本地代码路径。
    """
    return json.dumps({
        "status": "not_implemented",
        "message": f"adapt_generate_compat_layer 暂未接入（repo_path={repo_path!r}）",
        "tool": "adapt_generate_compat_layer",
    }, ensure_ascii=False)


def get_tools() -> list:
    return [adapt_review_repo, adapt_generate_compat_layer]
