from __future__ import annotations
"""LLM 构建工厂 — Anthropic Claude。"""


import asyncio
import json
import os
from functools import wraps
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic

from server.config import settings
from engine.exceptions import LLMAuthError, LLMNotFoundError, LLMQuotaError, LLMThinkingError


def retry_llm_call(max_retries: int = 3, base_delay: float = 1.0):
    """指数退避重试装饰器，用于 LLM 调用。"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        import time
                        delay = base_delay * (2 ** attempt)
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def _load_claude_sdk_env() -> dict[str, Any]:	 
    """读取 Claude SDK 配置文件 ~/.claude/settings.json 中的 env 配置。"""	 
    settings_path = Path.home() / ".claude" / "settings.json"	 
    if settings_path.exists():	 
        try: 
            data = json.loads(settings_path.read_text(encoding="utf-8")) 
            return data.get("env", {}) 
        except Exception: 
            pass 
    return {} 
 
 
 
 
def _resolve_anthropic_config() -> tuple[str, str | None, str | None]:	 
    """解析 Anthropic 配置，优先级：环境变量 > Claude SDK 配置 > 系统环境变量。	 


    当前环境使用 ANTHROPIC_AUTH_TOKEN 作为认证凭证。 


    Returns: 
        (api_key, base_url, model) 
    """ 
    claude_env = _load_claude_sdk_env() 


    api_key = ( 
        settings.anthropic_auth_token 
        or claude_env.get("ANTHROPIC_AUTH_TOKEN") 
        or os.getenv("ANTHROPIC_AUTH_TOKEN") 
        or "" 
    ) 
    base_url = ( 
        settings.anthropic_base_url 
        or claude_env.get("ANTHROPIC_BASE_URL") 
        or os.getenv("ANTHROPIC_BASE_URL") 
        or None 
    ) 
    model = ( 
        settings.anthropic_model 
        or claude_env.get("ANTHROPIC_MODEL") 
        or os.getenv("ANTHROPIC_MODEL") 
        or None 
    ) 
    return api_key, base_url, model 
 
 
 
 
def get_llm_model_name() -> str: 
    """返回当前实际使用的 LLM 模型名。""" 
    _, _, model = _resolve_anthropic_config() 
    return model or "claude-3-5-sonnet-20241022" 
 
 
 
 
def extract_text_content(content: str | list[Any]) -> str: 
    """从 LangChain Message content 中提取文本。 


    ChatAnthropic 返回的 content 是内容块列表，如 
    ``[{"type": "text", "text": "..."}]``，需要提取拼接。 
    """ 
    if isinstance(content, str): 
        return content 
    if isinstance(content, list): 
        parts: list[str] = [] 
        for block in content: 
            if isinstance(block, dict) and block.get("type") == "text": 
                parts.append(block.get("text", "")) 
        return "".join(parts) 
    return "" 
 
 
 
 
def build_llm(*, streaming: bool = False) -> ChatAnthropic: 
    """从配置构建 ChatAnthropic 实例，配置与 Claude SDK 共用。""" 
    api_key, base_url, model = _resolve_anthropic_config() 
    if not api_key:	 
        raise LLMAuthError(	 
            "请配置 ANTHROPIC_AUTH_TOKEN（环境变量、.env 或 ~/.claude/settings.json）" 
        ) 


    kwargs: dict[str, Any] = {	 
        "model": model or "claude-3-5-sonnet-20241022",	 
        "anthropic_api_key": api_key,	 
        "streaming": streaming,	 
    } 
    if base_url: 
        kwargs["anthropic_api_url"] = base_url 

    kwargs["default_headers"] = {"Authorization": f"Bearer {api_key}"}
    return ChatAnthropic(**kwargs)


def build_workflow_llm() -> ChatAnthropic:
    """专用于工作流规划的 LLM 实例。

    可用更快/便宜的模型做规划决策，通过 ``workflow_llm_model`` 配置。
    为空时复用主 LLM 模型。
    """
    api_key, base_url, model = _resolve_anthropic_config()
    if not api_key:
        raise LLMAuthError(
            "请配置 ANTHROPIC_AUTH_TOKEN（环境变量、.env 或 ~/.claude/settings.json）"
        )

    workflow_model = settings.workflow_llm_model or model or "claude-3-5-sonnet-20241022"

    kwargs: dict[str, Any] = {
        "model": workflow_model,
        "anthropic_api_key": api_key,
        "temperature": 0.1,  # 低温度保证规划一致性
    }
    if base_url:
        kwargs["anthropic_api_url"] = base_url

    kwargs["default_headers"] = {"Authorization": f"Bearer {api_key}"}
    return ChatAnthropic(**kwargs)


def handle_api_error(e: Exception) -> None:
    """统一识别 API 错误。"""
    err = str(e).lower()
    if "401" in err or "invalid authentication" in err or "invalid x-api-key" in err:
        raise LLMAuthError("API 认证失败（401），请检查 ANTHROPIC_AUTH_TOKEN") from None
    if "404" in err or "not_found" in err:
        raise LLMNotFoundError("API 路径 404，请检查 ANTHROPIC_BASE_URL") from None
    if "429" in err or "rate limit" in err or "insufficient balance" in err:
        raise LLMQuotaError("账户额度不足或请求过于频繁（429）") from None
    if "reasoning_content" in err or "thinking" in err:
        raise LLMThinkingError("thinking 与工具调用冲突（400）") from None
    raise e
