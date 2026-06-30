"""全局 fixtures — 所有测试共享的 mock 和工具函数。

注意：不使用 pytest-mock（环境受限），用 unittest.mock 替代。
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# 在模块导入阶段即 mock lts_logger，防止收集测试时加载真实模块。
_lts_logger_mod = types.ModuleType("lts_logger")
_lts_logger_mod.lts_logger = MagicMock()
sys.modules.setdefault("lts_logger", _lts_logger_mod)


@pytest.fixture(autouse=True)
def _mock_lts_logger():
    """自动 mock lts_logger 模块 — 防止导入时触发 producer SDK 依赖。"""
    yield _lts_logger_mod


# ── langchain / langgraph mock (not installed on this system) ─────────
class _FakeChatAnthropic:
    """模拟 ChatAnthropic，保存构造参数为属性。"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


_fake_lc = MagicMock()
_fake_lc.ChatAnthropic = _FakeChatAnthropic
sys.modules.setdefault("langchain_anthropic", _fake_lc)

_fake_lg = MagicMock()
_fake_lg.prebuilt = MagicMock()
_fake_lg.prebuilt.create_react_agent = MagicMock(return_value=MagicMock())
_fake_lg.checkpoint = MagicMock()
_fake_lg.checkpoint.memory = MagicMock()
_fake_lg.checkpoint.memory.MemorySaver = MagicMock()
sys.modules.setdefault("langgraph", _fake_lg)
sys.modules.setdefault("langgraph.prebuilt", _fake_lg.prebuilt)
sys.modules.setdefault("langgraph.checkpoint", _fake_lg.checkpoint)
sys.modules.setdefault("langgraph.checkpoint.memory", _fake_lg.checkpoint.memory)


@pytest.fixture(autouse=True)
def _mock_claude_sdk_env():
    """自动 mock Claude SDK 配置文件读取 — 避免本地配置影响测试结果。"""
    with patch("app.core.llm_factory._load_claude_sdk_env", return_value={}) as p:
        yield p


@pytest.fixture(autouse=True)
def _mock_subprocess_run():
    """自动 mock subprocess.run — 防止意外执行系统命令。"""
    mock = MagicMock()
    mock.return_value.returncode = 0
    mock.return_value.stdout = ""
    mock.return_value.stderr = ""
    with patch("subprocess.run", mock) as p:
        yield p


@pytest.fixture
def tmp_data_dir(tmp_path):
    """创建临时数据目录，模拟 ~/.mofix 结构。"""
    d = tmp_path / ".mofix"
    d.mkdir()
    return d
