"""
回测引擎,简化版交易环境封装。

- ``Portfolio``     维护持仓 / cash / 盈亏 / 撮合;含 ``from_scenario`` 工厂
- ``observation``   把状态渲染成 agent 可见的 markdown
- ``single_day``    单日 episode 编排 ``run_backtest_episode``
- ``multi_day``     多日连续回测,portfolio 跨天 persist,最后一天才清盘

前瞻收益评估(``compute_forward_return``)放在顶层 ``trade_agent.forward_return``,
因为它是 RL reward 的输入,不属于回测引擎内核 — backtest 跑完得到 portfolio 末态
后,由调用方(training workflow 或 dashboard)决定是否再算前瞻收益。
"""

from .multi_day import (
    DayResult,
    MultiDayResult,
    run_multi_day_backtest,
)
from .observation import (
    format_observation,
    format_portfolio_stats,
)
from .portfolio import Portfolio
from .single_day import run_backtest_episode
from schemas import EpisodeResult, Scenario

__all__ = [
    "Portfolio",
    "Scenario",
    "EpisodeResult",
    "run_backtest_episode",
    "run_multi_day_backtest",
    "DayResult",
    "MultiDayResult",
    "format_observation",
    "format_portfolio_stats",
]
