"""
OpenAI Agents SDK 接入层
"""

from __future__ import annotations

from typing import Literal

from agents import Agent, ModelSettings
from openai.types.shared.reasoning import Reasoning

from .prompts import DEFAULT_SYSTEM_PROMPT
from .tools import (
    compute_alpha_factor,
    get_macro_indicators,
    get_price_data,
    take_action,
    trade_agent_tools,
)
from schemas import TradeContext

ReasoningEffort = Literal["low", "medium", "high"]


def build_trade_agent(
    instructions: str = DEFAULT_SYSTEM_PROMPT,
    enable_thinking: bool = True,
    reasoning_effort: ReasoningEffort = "medium",
) -> Agent[TradeContext]:
    """
    构造挂好 4 个工具的交易 Agent

    1. get_price_data: 查询单只股票的历史行情和技术指标
    2. get_macro_indicators: 查询大盘指数走势
    3. compute_alpha_factor: 用 Qlib 表达式计算自定义 alpha 因子
    4. TakeAction: 执行买入/卖出/结束当日交易(type ∈ {buy, sell, stop})

    此处 Builder 是使用 AReaL 的标准写法,参考自 areal/workflow/openai_agent/math_agent.py 的
    build_math_agent 函数。

    ``enable_thinking=True`` 通过 ``ModelSettings(reasoning=Reasoning(...))`` 让 SDK 知道
    目标是 reasoning 模型(如 vLLM ``--reasoning-parser qwen3`` 部署的 Qwen3.5),SDK 会
    自动处理 ``message.reasoning_content`` 字段的多轮透传 + 流式 reasoning 事件
    (``response.reasoning_text.delta``)。参考 OpenAI Agents SDK 例子:
    ``examples/reasoning_content/gpt_oss_stream.py``。
    """
    reasoning = (
        Reasoning(effort=reasoning_effort, summary="detailed") if enable_thinking else None
    )
    return Agent[TradeContext](
        name="trade_agent",
        instructions=instructions,
        tools=trade_agent_tools,
        model_settings=ModelSettings(
            parallel_tool_calls=False,
            reasoning=reasoning,
        ),
    )


__all__ = [
    "TradeContext",
    "build_trade_agent",
    "trade_agent_tools",
    "get_price_data",
    "get_macro_indicators",
    "compute_alpha_factor",
    "take_action",
]
