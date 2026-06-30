"""本地专用 fixtures — 需要真实 LLM/外部服务。仅在本地手动运行，CI 中跳过。"""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests needing real LLM/external services")
