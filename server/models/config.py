"""配置修改接口数据模型。"""

from __future__ import annotations

from pydantic import BaseModel


class UpdateModelConfigRequest(BaseModel):
    """修改模型 API 配置请求体。"""

    access_token: str  # 对应 ANTHROPIC_AUTH_TOKEN
    base_url: str      # 对应 ANTHROPIC_BASE_URL
    model: str         # 对应 ANTHROPIC_MODEL
