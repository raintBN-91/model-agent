"""A 股交易领域数据类型与常量。无 I/O,仅 OpenAI Agents SDK 类型注解依赖。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backtest.portfolio import Portfolio


# 股票池约定为主板非 ST,涨跌停统一 ±10%。
PRICE_LIMIT_PCT: float = 0.10


@dataclass
class EnvConfig:
    """回测与训练环境配置。撮合参数与 Qlib ``exchange_kwargs`` 对齐。"""

    qlib_data_dir: str = "data/qlib_data"
    initial_cash: float = 1_000_000.0

    # 撮合:开仓费率/平仓费率/最低手续费/滑点。
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_cost: float = 5.0
    slippage_rate: float = 0.0005

    # 市场状态计算器参考的指数(供外部 dashboard 使用)。
    index: str = "SH000001"
    volatility_lookback: int = 252
    large_cap_index: str = "SH000300"
    small_cap_index: str = "SH000905"

    # Agent 行为上限。
    max_actions_per_day: int = 10
    universe: list[str] = field(default_factory=list)


@dataclass
class Position:
    """单只股票的持仓状态。``buy_date`` 记录最近一次买入日,用于 T+1 校验。"""

    symbol: str
    shares: int
    cost_basis: float   # 含滑点的加权平均成本
    current_price: float
    buy_date: str = ""

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.cost_basis) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.current_price / self.cost_basis - 1


@dataclass
class TradeRecord:
    """一笔成交记录。``price`` 是含滑点的实际成交价。"""

    date: str
    symbol: str
    direction: str   # "buy" | "sell"
    shares: int
    price: float
    commission: float
    slippage_cost: float
    reasoning: str = ""


@dataclass
class MarketState:
    """单个交易日的全市场标签快照,由 dashboard 端的计算器生成。"""

    date: str
    volatility_label: str
    trend_short_label: str
    trend_mid_label: str
    breadth_label: str
    activity_label: str
    style_label: str
    volatility_percentile: float
    momentum_5d: float
    momentum_20d: float
    advance_ratio: float
    volume_deviation: float
    style_excess: float

    def to_text(self) -> str:
        return (
            f"## 当前市场状态（{self.date}）\n\n"
            f"- 波动率环境：{self.volatility_label}"
            f"（近20日波动率处于过去一年的 {self.volatility_percentile:.0f} 分位）\n"
            f"- 市场趋势：短期{self.trend_short_label}（5日动量 {self.momentum_5d:+.1f}%），"
            f"中期{self.trend_mid_label}（20日动量 {self.momentum_20d:+.1f}%）\n"
            f"- 市场宽度：{self.breadth_label}（上涨股票占比 {self.advance_ratio:.0%}）\n"
            f"- 成交活跃度：{self.activity_label}（近5日成交额较20日均值"
            f"{'上升' if self.volume_deviation > 0 else '下降'} "
            f"{abs(self.volume_deviation):.0f}%）\n"
            f"- 风格倾向：{self.style_label}（近20日超额 {self.style_excess:+.1f}%）"
        )


@dataclass
class Scenario:
    """单个交易日 backtest 的输入定义。"""

    start_date: str
    universe: list[str]
    initial_cash: float = 1_000_000.0
    initial_positions: list[dict[str, Any]] = field(default_factory=list)
    """每项形如 ``{"symbol": str, "shares": int, "price": float}``。"""
    market_state_text: str = ""
    env_id: str = ""
    """自由格式的场景标识,会进入日志/审计记录。"""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scenario:
        """从 dataset row(可能含 datetime/Timestamp 类型的 ``start_date``)解析出 Scenario。"""
        return cls(
            start_date=normalize_date(d["start_date"]),
            universe=list(d.get("universe") or []),
            initial_cash=float(d.get("initial_cash", 1_000_000.0)),
            initial_positions=list(d.get("initial_positions") or []),
            market_state_text=str(d.get("market_state_text", "")),
            env_id=str(d.get("env_id", "")),
        )


@dataclass
class EpisodeResult:
    """单次 episode 完成(或中断)后的返回值。"""

    scenario: Scenario
    portfolio: Portfolio
    trades: list[TradeRecord]
    final_positions: dict[str, Position]
    audit_log: list[dict]
    metrics: dict[str, Any]
    """``compute_forward_return`` 的输出;失败回退到 ``EMPTY_METRICS``。"""
    num_turns: int = 0


@dataclass
class TradeContext:
    """Episode 共享的可变运行时状态,通过 ``RunContextWrapper`` 传给每次工具调用。"""

    portfolio: Portfolio
    current_date: str
    universe: list[str]
    max_actions_per_day: int

    # 可变计数器与标记 —— 工具调用时就地修改。
    actions_this_day: int = 0
    done: bool = False

    # ``done`` 翻成 True 当下、``liquidate_all()`` 调用之前的持仓快照,
    # 供 ``compute_forward_return`` 计算前瞻收益。
    final_positions: dict[str, Position] | None = None

    # 审计日志:runner 填充,供 dashboard / 训练日志使用。
    audit_log: list[dict] = field(default_factory=list)


def normalize_date(val: Any) -> str:
    """将 datetime / pd.Timestamp / str 统一为 'YYYY-MM-DD' 字符串。"""
    if isinstance(val, str):
        return val
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d")
    if hasattr(val, "strftime"):  # pd.Timestamp 等
        return val.strftime("%Y-%m-%d")
    return str(val) if val else ""


__all__ = [
    "PRICE_LIMIT_PCT",
    "EnvConfig",
    "Position",
    "TradeRecord",
    "MarketState",
    "Scenario",
    "EpisodeResult",
    "TradeContext",
    "normalize_date",
]
