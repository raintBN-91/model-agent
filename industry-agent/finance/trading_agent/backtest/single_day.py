"""单日 backtest episode 编排:Agent ↔ Portfolio。

无框架依赖,只要客户端实现 ``openai.AsyncOpenAI`` 接口即可:
训练时传 AReaL 的 ``ArealOpenAI``(回传 token logprob),
离线回测/dashboard 时传标准 ``AsyncOpenAI`` 指向 vLLM 即可。
多日连续回测见 ``trade_agent.backtest.multi_day.run_multi_day_backtest``。
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

from agents import Agent, OpenAIProvider, RunConfig, Runner
from agents.exceptions import (
    AgentsException,
    MaxTurnsExceeded,
    ModelBehaviorError,
)

from agent.prompts import render_user_prompt
from .observation import format_portfolio_stats
from .portfolio import Portfolio
from data.qlib_client import ensure_qlib, update_portfolio_prices
from schemas import EnvConfig, EpisodeResult, Scenario, TradeContext

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _finalize(state: TradeContext, liquidate: bool) -> None:
    """收尾:可选清盘。

    ``liquidate=True``(单日 episode 默认): snapshot 持仓 → 强制清盘。``final_positions`` 是
    清盘前 deep copy 的快照,供 caller 算前瞻收益用(见 ``training/forward_return.py``)。
    ``liquidate=False``(多日跑的中间日): 持仓自然带到下一天。
    """
    if not state.done:
        # Agent 没有显式调用 TakeAction(type="stop"),自动结束。
        state.done = True

    if liquidate and state.final_positions is None:
        update_portfolio_prices(state.portfolio, state.current_date)
        state.final_positions = copy.deepcopy(dict(state.portfolio.positions))
        state.portfolio.liquidate_all(state.current_date)


async def run_backtest_episode(
    scenario: Scenario,
    agent: Agent[TradeContext],
    openai_client: AsyncOpenAI,
    *,
    env_config: EnvConfig | None = None,
    max_turns: int = 20,
    model_settings: Any = None,
    model: str | None = None,
    liquidate: bool = True,
) -> EpisodeResult:
    """跑完一个交易日 episode 并返回结果。

    推理回测用 ``openai.AsyncOpenAI``,训练中用 ``ArealOpenAI``;两个都满足
    AsyncOpenAI 接口。Qlib 数据目录从 ``env_config.qlib_data_dir`` 读;
    ``ensure_qlib`` 是幂等的。

    返回的 ``EpisodeResult.metrics`` 是空 dict 占位,backtest 不算评估指标
    (那是 RL reward 设计的一部分,见 ``trade_agent.training.forward_return``);
    调用方拿到 ``portfolio`` + ``final_positions`` 后自己决定怎么打分。
    """
    cfg = env_config or EnvConfig()
    ensure_qlib(cfg.qlib_data_dir)

    portfolio = Portfolio.from_scenario(scenario, cfg)
    max_actions = len(scenario.universe) if scenario.universe else cfg.max_actions_per_day

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

    # 固定走 /v1/chat/completions: AReaL ArealOpenAI 是 chat completions 模式;
    # vLLM Qwen3 reasoning 通过 message.reasoning_content 字段返回。
    # model_settings=None 时 Agent.model_settings 生效(含 reasoning + parallel_tool_calls=False)。
    run_config = RunConfig(
        model=model or "default",
        model_provider=OpenAIProvider(openai_client=openai_client, use_responses=False),
        tracing_disabled=True,
        model_settings=model_settings,
    )

    # RL 训练里如果模型 hallucinate 一个不存在的 tool 名,agents SDK 的
    # process_model_response 默认 raise ModelBehaviorError,整个 episode 抛出 →
    # workflow 失败 → batch 永远凑不齐,actor 训不到第一个 step。
    #
    # 这里把它当成 episode 早结束:portfolio 已持有的仓位被强制清盘,reward 走
    # 默认负值路径(forward_return 算出来通常是 0 或负,经 reward_fn 离散化后
    # 是 -3 ~ 0)。模型在 RL 里学到 "调错 tool 名 → 负 reward"。
    #
    # MaxTurnsExceeded 一并 catch:agent 跑满 max_turns 没 stop 也是合法收尾。
    try:
        result = await Runner.run(
            agent,
            input=user_prompt,
            context=ctx,
            max_turns=max_turns,
            run_config=run_config,
        )
        raw_responses = getattr(result, "raw_responses", []) or []
    except (ModelBehaviorError, MaxTurnsExceeded) as e:
        logger.warning(
            "Episode aborted by agents SDK (date=%s, env_id=%s): %s",
            scenario.start_date, scenario.env_id, e,
        )
        raw_responses = []
    except AgentsException as e:
        # 其他 agents SDK 异常也兜底(如 InvalidJSONError 等)
        logger.warning(
            "Episode aborted by AgentsException (date=%s, env_id=%s): %s",
            scenario.start_date, scenario.env_id, e,
        )
        raw_responses = []

    _finalize(ctx, liquidate=liquidate)

    # raw_responses 是每次 LLM 调用一条,对应 "agent turn";new_items 还含 tool 消息会膨胀。
    num_turns = len(raw_responses)

    return EpisodeResult(
        scenario=scenario,
        portfolio=portfolio,
        trades=list(portfolio.trade_history),
        final_positions=ctx.final_positions or {},
        audit_log=list(ctx.audit_log),
        metrics={},
        num_turns=num_turns,
    )
