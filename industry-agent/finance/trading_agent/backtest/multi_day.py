"""跨多日连续回测 runner。

跟单日 ``run_backtest_episode`` 的区别:
1. **Portfolio 跨天 persist**:第 N 天结束时不清盘,持仓自然带到第 N+1 天
2. **每天独立 conversation**:每天 ``Runner.run`` 是新 conversation
3. **最终一次清盘**:跑完所有交易日才 liquidate + 算 r_outcome
4. **每日 NAV 曲线**:返回 daily_results 含 NAV 变化,可画曲线/算 Sharpe

适用场景:dashboard 多日回测,验证模型决策的连续性表现。
训练管道仍用单日 ``run_backtest_episode`` —— 每个 RL episode = 一个交易日。
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agents import OpenAIProvider, RunConfig, Runner

from agent.prompts import render_user_prompt
from .observation import format_portfolio_stats
from .portfolio import Portfolio
from data.qlib_client import ensure_qlib, update_portfolio_prices
from schemas import EnvConfig, Scenario, TradeContext, TradeRecord

if TYPE_CHECKING:
    from agents import Agent
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class DayResult:
    """一个交易日的结果(多日跑里的一行)。"""

    date: str
    nav_before: float
    nav_after: float
    daily_return: float
    trades_this_day: list[TradeRecord] = field(default_factory=list)
    audit_log: list[dict] = field(default_factory=list)
    num_turns: int = 0


@dataclass
class MultiDayResult:
    """跨多日连续回测的完整结果。

    ``metrics`` 占位空 dict,backtest 不算评估指标。caller 拿到 ``portfolio`` +
    ``final_positions`` 后自己调 ``trade_agent.training.forward_return.compute_forward_return``
    算前瞻收益,或者从 ``nav_curve`` 算 Sharpe / max drawdown 等经典指标。
    """

    portfolio: Portfolio        # 最后状态(已清盘)
    final_positions: dict       # 清盘前的最终持仓快照
    nav_curve: list[float]      # 长度 = len(scenarios) + 1, 含起点
    daily_results: list[DayResult]
    metrics: dict[str, Any]     # 占位空 dict,由 caller 填


async def run_multi_day_backtest(
    scenarios: list[Scenario],
    agent: Agent[TradeContext],
    openai_client: AsyncOpenAI,
    *,
    env_config: EnvConfig | None = None,
    max_turns_per_day: int = 20,
    model_settings: Any = None,
    model: str | None = None,
) -> MultiDayResult:
    """对一组 Scenario 按日期顺序逐日跑,portfolio 跨天 persist。

    ``scenarios[0].initial_cash`` / ``initial_positions`` 用作初始状态;
    后续 scenario 的这两个字段被忽略(portfolio 已 carry over)。
    跨天用同一个 agent 实例,每天新建 ``TradeContext`` reset 计数器。
    """
    if not scenarios:
        raise ValueError("scenarios 不能为空")

    cfg = env_config or EnvConfig()
    ensure_qlib(cfg.qlib_data_dir)

    # 1. 用第一个 scenario 初始化 portfolio(只用 initial_cash + initial_positions)。
    portfolio = Portfolio.from_scenario(scenarios[0], cfg)
    nav_curve: list[float] = [portfolio.get_nav()]
    daily_results: list[DayResult] = []

    # RunConfig 复用 — model_provider / model_settings 跨天共享。
    run_config = RunConfig(
        model=model or "default",
        model_provider=OpenAIProvider(openai_client=openai_client, use_responses=False),
        tracing_disabled=True,
        model_settings=model_settings,
    )

    # 2. 逐日跑。
    for scenario in scenarios:
        nav_before = portfolio.get_nav()
        trades_before = len(portfolio.trade_history)

        # 收盘价更新到当天(prev_close 等用)。
        update_portfolio_prices(portfolio, scenario.start_date)

        max_actions = (
            len(scenario.universe) if scenario.universe else cfg.max_actions_per_day
        )
        ctx = TradeContext(
            portfolio=portfolio,
            current_date=scenario.start_date,
            universe=list(scenario.universe),
            max_actions_per_day=max_actions,
        )

        user_prompt = render_user_prompt(
            current_date=scenario.start_date,
            portfolio_text=format_portfolio_stats(portfolio),
            universe=scenario.universe,
            market_state_text=scenario.market_state_text,
        )

        result = await Runner.run(
            agent,
            input=user_prompt,
            context=ctx,
            max_turns=max_turns_per_day,
            run_config=run_config,
        )

        # 当日不清盘,只更新现价 → 算 NAV → 进下一天。
        update_portfolio_prices(portfolio, scenario.start_date)
        nav_after = portfolio.get_nav()
        nav_curve.append(nav_after)

        num_turns = len(getattr(result, "raw_responses", []) or [])
        daily_trades = list(portfolio.trade_history[trades_before:])
        daily_results.append(DayResult(
            date=scenario.start_date,
            nav_before=nav_before,
            nav_after=nav_after,
            daily_return=(nav_after / nav_before - 1) if nav_before > 0 else 0.0,
            trades_this_day=daily_trades,
            audit_log=list(ctx.audit_log),
            num_turns=num_turns,
        ))

    # 3. 最后一天才清盘。评估指标由 caller 算(用 portfolio + final_positions)。
    last_date = scenarios[-1].start_date
    update_portfolio_prices(portfolio, last_date)
    final_positions = copy.deepcopy(dict(portfolio.positions))
    portfolio.liquidate_all(last_date)

    return MultiDayResult(
        portfolio=portfolio,
        final_positions=final_positions,
        nav_curve=nav_curve,
        daily_results=daily_results,
        metrics={},
    )
