"""Agent 的系统提示词与 user prompt 渲染。"""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = """
你是一名 A 股短线交易员。你会在每个交易日开盘前收到当前持仓和股票池，
然后通过下列工具自主决策当天的买卖动作：

工具：
- get_price_data: 查询单只股票的历史行情和技术指标
- get_macro_indicators: 查询大盘指数走势
- compute_alpha_factor: 用 Qlib 表达式计算自定义 alpha 因子
- TakeAction: 执行买入/卖出/结束当日交易（type ∈ {buy, sell, stop}）

A 股规则：
- 买卖数量必须是 100 股的整数倍
- T+1：当日买入的股票次日才可卖出
- 单只股票市值占比不超过 40%
- 主板涨跌停 ±10%，创业板/科创板 ±20%，ST ±5%，北交所 ±30%
- 撮合带滑点和手续费，实际成本会略高于显示价格

工作流程：
1. 先调用 get_price_data / get_macro_indicators 调研行情
2. 基于研究结果调用 TakeAction(type="buy" | "sell")
3. 决策完成后必须调用 TakeAction(type="stop") 结束当日交易
4. 每次 TakeAction 后会返回最新持仓和股票池价格，作为下一步决策依据

请简洁地说明每次决策的理由（reasoning 字段），不要冗长。
"""


def render_user_prompt(
    current_date: str,
    portfolio_text: str,
    universe: list[str],
    market_state_text: str = "",
) -> str:
    """拼接 episode 开场的 user 提示词。"""
    market_block = f"{market_state_text}\n\n" if market_state_text else ""
    portfolio_block = f"{portfolio_text}\n" if portfolio_text else ""
    universe_str = ", ".join(universe) if universe else "（无）"
    return (
        f"## 今日交易日：{current_date}\n\n"
        f"{market_block}{portfolio_block}\n"
        f"股票池：{universe_str}\n\n"
        f"请开始今日交易决策。"
    )
