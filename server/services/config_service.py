"""模型配置修改服务 — 同步更新文件与运行时内存。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from server.config import settings
from engine.exceptions import MoFixException

logger = logging.getLogger(__name__)


def get_claude_settings_path() -> Path:
    """返回 Claude SDK 配置文件路径。"""
    return Path.home() / ".claude" / "settings.json"


def load_claude_settings() -> dict[str, Any]:
    """读取 ~/.claude/settings.json，失败时返回空字典。"""
    path = get_claude_settings_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("读取 Claude 配置文件失败，将覆盖写入: %s", e)
    return {}


def save_claude_settings(data: dict[str, Any]) -> None:
    """保存配置到 ~/.claude/settings.json，必要时自动创建目录。"""
    path = get_claude_settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.exception("写入 Claude 配置文件失败")
        raise MoFixException(f"无法保存 Claude 配置文件: {e}") from None


def update_model_config(access_token: str, base_url: str, model: str) -> bool:
    """更新 Anthropic 相关配置到文件和 settings 单例。

    Args:
        access_token: 对应 ANTHROPIC_AUTH_TOKEN。
        base_url: 对应 ANTHROPIC_BASE_URL。
        model: 对应 ANTHROPIC_MODEL。

    Returns:
        是否成功。
    """
    data = load_claude_settings()
    if "env" not in data or not isinstance(data["env"], dict):
        data["env"] = {}

    data["env"]["ANTHROPIC_AUTH_TOKEN"] = access_token
    data["env"]["ANTHROPIC_BASE_URL"] = base_url
    data["env"]["ANTHROPIC_MODEL"] = model

    save_claude_settings(data)

    # 同步更新运行时内存，确保当前进程立即生效
    settings.anthropic_auth_token = access_token
    settings.anthropic_base_url = base_url
    settings.anthropic_model = model

    logger.info("已更新 Anthropic 模型配置: model=%s, base_url=%s", model, base_url)
    return True
