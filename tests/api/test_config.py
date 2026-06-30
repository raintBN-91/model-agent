"""测试 POST /v1/chat/config 配置修改接口。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import config_service


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def patched_settings_path(tmp_path, monkeypatch):
    """将 Claude 配置文件重定向到临时目录，避免污染真实文件。"""
    path = tmp_path / "claude" / "settings.json"

    def _fake_path():
        return path

    monkeypatch.setattr(config_service, "get_claude_settings_path", _fake_path)
    return path


def test_update_model_config_success(client: TestClient, patched_settings_path):
    """正常更新配置文件和运行时内存。"""
    response = client.post(
        "/v1/chat/config",
        json={
            "access_token": "new-token",
            "base_url": "https://new.example.com",
            "model": "claude-new-model",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"success": True}

    # 验证文件内容
    data = json.loads(patched_settings_path.read_text(encoding="utf-8"))
    assert data["env"]["ANTHROPIC_AUTH_TOKEN"] == "new-token"
    assert data["env"]["ANTHROPIC_BASE_URL"] == "https://new.example.com"
    assert data["env"]["ANTHROPIC_MODEL"] == "claude-new-model"

    # 验证运行时内存已同步
    assert settings.anthropic_auth_token == "new-token"
    assert settings.anthropic_base_url == "https://new.example.com"
    assert settings.anthropic_model == "claude-new-model"


def test_update_model_config_creates_file(client: TestClient, patched_settings_path):
    """配置文件不存在时自动创建目录和文件。"""
    assert not patched_settings_path.exists()

    response = client.post(
        "/v1/chat/config",
        json={
            "access_token": "token",
            "base_url": "",
            "model": "",
        },
    )
    assert response.status_code == 200
    assert patched_settings_path.exists()

    data = json.loads(patched_settings_path.read_text(encoding="utf-8"))
    assert data["env"]["ANTHROPIC_AUTH_TOKEN"] == "token"
    assert data["env"]["ANTHROPIC_BASE_URL"] == ""
    assert data["env"]["ANTHROPIC_MODEL"] == ""


def test_update_model_config_preserves_other_fields(
    client: TestClient, patched_settings_path
):
    """更新时保留文件中其他字段。"""
    patched_settings_path.parent.mkdir(parents=True, exist_ok=True)
    patched_settings_path.write_text(
        json.dumps({"env": {"OTHER_KEY": "other-value"}, "keep": True}),
        encoding="utf-8",
    )

    response = client.post(
        "/v1/chat/config",
        json={
            "access_token": "token",
            "base_url": "http://localhost",
            "model": "model-x",
        },
    )
    assert response.status_code == 200

    data = json.loads(patched_settings_path.read_text(encoding="utf-8"))
    assert data["env"]["OTHER_KEY"] == "other-value"
    assert data["keep"] is True


def test_update_model_config_overwrites_corrupted_file(
    client: TestClient, patched_settings_path
):
    """JSON 损坏时覆盖写入。"""
    patched_settings_path.parent.mkdir(parents=True, exist_ok=True)
    patched_settings_path.write_text("not-json", encoding="utf-8")

    response = client.post(
        "/v1/chat/config",
        json={
            "access_token": "token",
            "base_url": "https://x.com",
            "model": "model-y",
        },
    )
    assert response.status_code == 200

    data = json.loads(patched_settings_path.read_text(encoding="utf-8"))
    assert data["env"]["ANTHROPIC_AUTH_TOKEN"] == "token"


def test_update_model_config_missing_field(client: TestClient):
    """缺少必填字段返回 422。"""
    response = client.post("/v1/chat/config", json={"access_token": "token"})
    assert response.status_code == 422
