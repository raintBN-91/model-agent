"""
通过 OpenAI Agents SDK @function_tool 定义的交易工具。

共享状态(Portfolio / current_date / universe)放在 TradeContext,通过 RunContextWrapper.context 访问

SDK 把同一 context 对象传给单次 Runner.run 内的每次工具调用。工具只返回字符串；清盘、reward、指标日志由 runner/workflow 负责。
"""

from __future__ import annotations

import concurrent.futures
import logging
import re

import numpy as np
import pandas as pd
from agents import RunContextWrapper, Tool, function_tool

from backtest.observation import format_observation
from data import format_date, format_volume
from data.indicators import (
    BASE_FIELDS,
    EXPRESSION_PATTERN,
    INDICATOR_MAP,
    compute_rsi_python,
    parse_period_days,
    resolve_start_date,
)
from data.qlib_client import (
    QLIB_TIMEOUT,
    get_current_price,
    get_prev_close,
    safe_features,
)
from schemas import TradeContext

logger = logging.getLogger(__name__)


# =============================================================================
# 只读工具:行情 / 指数 / 自定义因子
# =============================================================================


@function_tool(name_override="get_price_data")
def get_price_data(
    ctx: RunContextWrapper[TradeContext],
    symbol: str,
    period: str,
    indicators: list[str] | None = None,
) -> str:
    """查询单只股票的历史 OHLCV 与技术指标。

    Args:
        symbol: A 股代码,例如 ``SH600519``。
        period: 回看窗口,例如 ``5d``、``20d``、``60d``。
        indicators: 指标名,取自支持集合
            (MA5/MA10/MA20/MA60/EMA12/EMA26/VOL_20/ROC_5/ROC_20/VWAP/
            VOL_RATIO_5_20/MACD_DIF/BOLL_UPPER/BOLL_LOWER/RSI_14);
            其他自定义因子请用 compute_alpha_factor。
    """
    return _query_price_data(ctx.context.current_date, symbol, period, indicators or [])


def _query_price_data(
    current_date: str,
    symbol: str,
    period: str,
    indicators: list[str],
) -> str:
    """``get_price_data`` 的纯函数实现,供 dashboard / 单测直接调用。失败返回 ``[错误] ...`` 字符串。"""
    # 1. 解析回看窗口,带 buffer 覆盖非交易日。
    period_days = parse_period_days(period)
    if period_days is None:
        return f"[错误] 不支持的 period 格式: {period}"
    start_date = resolve_start_date(current_date, period_days)

    # 2. 组装 Qlib 字段:基础 OHLCV + 用户指定的技术指标。
    fields = list(BASE_FIELDS)
    columns = ["open", "high", "low", "close", "volume"]
    rsi_needed = False
    for name in indicators:
        upper = name.upper()
        if upper not in INDICATOR_MAP:
            return f"[错误] 未知指标 '{name}',请使用 compute_alpha_factor 自定义表达式"
        fields.append(INDICATOR_MAP[upper])
        columns.append(name)
        rsi_needed = rsi_needed or upper.startswith("RSI")

    # 3. 查 Qlib(带超时保护),整理成单只股票的时间序列。
    try:
        df = safe_features([symbol], fields, start_time=start_date, end_time=current_date)
    except concurrent.futures.TimeoutError:
        return f"[错误] 获取 {symbol} 数据超时 ({QLIB_TIMEOUT}s)"
    except Exception as e:  # noqa: BLE001
        return f"[错误] 获取 {symbol} 数据失败: {e}"
    if df.empty:
        return f"[错误] {symbol} 在 {start_date} ~ {current_date} 无数据"
    df.columns = columns
    if isinstance(df.index, pd.MultiIndex):
        if symbol not in df.index.get_level_values(0):
            return f"[错误] {symbol} 无数据"
        df = df.xs(symbol, level=0)

    # 4. RSI 容错:Qlib 表达式在数据稀疏时会出 inf/nan,用 Python SMA 兜底。
    if rsi_needed:
        for col in (c for c in df.columns if c.upper().startswith("RSI")):
            if df[col].isna().all() or np.isinf(df[col]).any():
                logger.warning("RSI Qlib 表达式失败,用 Python SMA fallback")
                m = re.search(r"(\d+)", col)
                df[col] = compute_rsi_python(df["close"], int(m.group(1)) if m else 14)

    df = df.tail(period_days)

    # 5. 格式化输出:最新 OHLCV → 区间统计 → 指标值 → 近 5 日走势。
    latest = df.iloc[-1]
    lines = [f"## {symbol} 近{len(df)}个交易日行情\n"]
    lines.append(
        f"最新:{format_date(df.index[-1])} "
        f"开{latest['open']:.2f} 高{latest['high']:.2f} 低{latest['low']:.2f} "
        f"收{latest['close']:.2f} 量{format_volume(float(latest.get('volume') or 0))}"
    )
    lines.append(
        f"区间:最高{df['high'].max():.2f}({format_date(df['high'].idxmax())}) "
        f"最低{df['low'].min():.2f}({format_date(df['low'].idxmin())}) "
        f"涨跌幅{df['close'].iloc[-1] / df['close'].iloc[0] - 1:+.1%}"
    )

    indicator_cols = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
    if indicator_cols:
        lines.append("\n技术指标(最新值):")
        for col in indicator_cols:
            val = latest[col]
            lines.append(f"- {col}: {val:.2f}" if pd.notna(val) else f"- {col}: N/A")

    lines.append(f"\n近{min(5, len(df))}日走势:")
    prev_close = None
    for date, row in df.tail(5).iterrows():
        arrow = ""
        if prev_close is not None:
            arrow = " ▲" if row["close"] > prev_close else " ▼" if row["close"] < prev_close else " ─"
        lines.append(
            f"{format_date(date)}: 收{row['close']:.2f} "
            f"量{format_volume(float(row.get('volume') or 0))}{arrow}"
        )
        prev_close = row["close"]

    return "\n".join(lines)


@function_tool(name_override="get_macro_indicators")
def get_macro_indicators(
    ctx: RunContextWrapper[TradeContext],
    index: str = "SH000300",
    period: str = "20d",
) -> str:
    """查询大盘指数近期走势。

    Args:
        index: 指数代码,例如 ``SH000300``(沪深 300)、``SH000001``(上证综指)。
        period: 回看窗口,例如 ``5d``、``20d``。
    """
    return _query_macro_indicators(ctx.context.current_date, index, period)


def _query_macro_indicators(current_date: str, index: str, period: str) -> str:
    """``get_macro_indicators`` 的纯函数实现,供 dashboard / 单测直接调用。"""
    # 1. 解析回看窗口。
    period_days = parse_period_days(period)
    if period_days is None:
        return f"[错误] 不支持的 period 格式: {period}"
    start_date = resolve_start_date(current_date, period_days)

    # 2. 查 Qlib,整理成单只指数的时间序列。
    try:
        df = safe_features([index], BASE_FIELDS, start_time=start_date, end_time=current_date)
    except concurrent.futures.TimeoutError:
        return f"[错误] 获取指数 {index} 数据超时 ({QLIB_TIMEOUT}s)"
    except Exception as e:  # noqa: BLE001
        return f"[错误] 获取指数 {index} 数据失败: {e}"
    if df.empty:
        return f"[错误] 指数 {index} 在 {start_date} ~ {current_date} 无数据"
    df.columns = ["open", "high", "low", "close", "volume"]
    if isinstance(df.index, pd.MultiIndex):
        if index not in df.index.get_level_values(0):
            return f"[错误] 指数 {index} 无数据"
        df = df.xs(index, level=0)
    df = df.tail(period_days)

    # 3. 格式化输出:最新 OHLCV → 区间收益与波动率 → 近 5 日走势。
    latest = df.iloc[-1]
    lines = [f"## 指数 {index} 近{len(df)}个交易日行情\n"]
    lines.append(
        f"最新:{format_date(df.index[-1])} "
        f"开{latest['open']:.2f} 高{latest['high']:.2f} 低{latest['low']:.2f} "
        f"收{latest['close']:.2f} 量{format_volume(float(latest.get('volume') or 0))}"
    )
    if len(df) > 1:
        period_return = df["close"].iloc[-1] / df["close"].iloc[0] - 1
        volatility = df["close"].pct_change().std()
        lines.append(f"区间涨跌幅:{period_return:+.1%},日波动率:{volatility:.3f}")

    lines.append(f"\n近{min(5, len(df))}日走势:")
    prev_close = None
    for date, row in df.tail(5).iterrows():
        arrow = ""
        if prev_close is not None:
            arrow = " ▲" if row["close"] > prev_close else " ▼" if row["close"] < prev_close else " ─"
        lines.append(
            f"{format_date(date)}: 收{row['close']:.2f} "
            f"量{format_volume(float(row.get('volume') or 0))}{arrow}"
        )
        prev_close = row["close"]

    return "\n".join(lines)


@function_tool(name_override="compute_alpha_factor")
def compute_alpha_factor(
    ctx: RunContextWrapper[TradeContext],
    expression: str,
    symbols: list[str],
    period: str = "60d",
) -> str:
    """对一组股票求一个自定义 Qlib 表达式。

    Args:
        expression: Qlib 表达式,例如 ``Corr($close, Log($volume), 10)``;
            仅接受 ``$open/$high/$low/$close/$volume`` 字段加 Qlib 算子
            (Mean、Std、Ref、EMA、Sum、If、Log、Corr 等)。
        symbols: 待评估的股票代码列表。
        period: 回看窗口,例如 ``60d``。
    """
    return _query_alpha_factor(ctx.context.current_date, expression, symbols, period)


def _query_alpha_factor(
    current_date: str,
    expression: str,
    symbols: list[str],
    period: str,
) -> str:
    """``compute_alpha_factor`` 的纯函数实现,供 dashboard / 单测直接调用。"""
    # 1. 安全校验:只接受 Qlib 操作符与变量符号。
    if not EXPRESSION_PATTERN.match(expression):
        return f"[错误] 表达式包含非法字符: {expression}"
    period_days = parse_period_days(period)
    if period_days is None:
        return f"[错误] 不支持的 period 格式: {period}"
    start_date = resolve_start_date(current_date, period_days)

    # 2. 查 Qlib 求表达式值。
    try:
        df = safe_features(symbols, [expression], start_time=start_date, end_time=current_date)
    except concurrent.futures.TimeoutError:
        return f"[错误] 表达式计算超时 ({QLIB_TIMEOUT}s)"
    except Exception as e:  # noqa: BLE001
        return f"[错误] 表达式计算失败: {e}"
    if df.empty:
        return "[错误] 计算结果为空"
    df.columns = ["factor"]
    df = df.tail(period_days * len(symbols))

    # 3. 取每只股票最新值。
    latest_vals: dict[str, float] = {}
    if isinstance(df.index, pd.MultiIndex):
        for instrument in df.index.get_level_values(0).unique():
            sub = df.xs(instrument, level=0)
            if not sub.empty and pd.notna(sub.iloc[-1, 0]):
                latest_vals[instrument] = float(sub.iloc[-1, 0])
    elif pd.notna(df["factor"].iloc[-1]):
        latest_vals = {symbols[0] if symbols else "unknown": float(df["factor"].iloc[-1])}

    # 4. 格式化输出:按因子值降序排名。
    lines = [f"## 因子: {expression}\n"]
    if not latest_vals:
        lines.append("无有效计算结果")
        return "\n".join(lines)
    lines.append("最新值排名:")
    for rank, (sym, val) in enumerate(
        sorted(latest_vals.items(), key=lambda x: x[1], reverse=True), 1
    ):
        lines.append(f"{rank}. {sym}: {val:+.4f}")
    return "\n".join(lines)


# =============================================================================
# 有状态工具:执行决策 + 收盘清盘
# =============================================================================


@function_tool(name_override="TakeAction")
def take_action(
    ctx: RunContextWrapper[TradeContext],
    type: str,
    symbol: str = "",
    amount: int = 0,
    reasoning: str = "",
) -> str:
    """执行一次交易决策并返回成交后的观测。

    Args:
        type: ``buy``、``sell``、``stop`` 之一;下完所有目标单后用 ``stop`` 结束当日。
        symbol: 股票代码;buy/sell 必填,stop 时忽略。
        amount: 股数;必须是 100 的整数倍,buy/sell 必填。
        reasoning: 决策理由(一句话),仅作审计,不影响撮合。
    """
    state = ctx.context

    # 1. 分发动作:tool 只改变 portfolio + 设置 done 标记,不负责清盘。
    #    清盘/snapshot 由 runner 层(_wrap_up_episode)决定 —— 单日 runner 会清盘,
    #    多日 runner 跨天不清盘,只在最后一天清。
    if type in ("buy", "sell"):
        _execute_market_order(state, type, symbol, amount, reasoning)
    elif type == "stop":
        state.done = True
        state.audit_log.append(
            {"date": state.current_date, "action": "stop", "reasoning": reasoning}
        )
    else:
        logger.warning("未知动作类型: %s", type)

    # 2. 返回观测给 agent。
    return format_observation(
        portfolio=state.portfolio,
        current_date=state.current_date,
        universe=state.universe,
        actions_this_day=state.actions_this_day,
        max_actions_per_day=state.max_actions_per_day,
        done=state.done,
    )


def _execute_market_order(
    state: TradeContext,
    direction: str,
    symbol: str,
    amount: int,
    reasoning: str,
) -> None:
    """校验 → 查价 → 下单 → 写审计。计数器无论成交与否都自增,防止刷拒单。"""

    # 1. 参数与股票池校验。
    if not symbol or amount == 0:
        logger.warning("%s 缺参数: symbol=%s amount=%s", direction, symbol, amount)
        return
    if state.universe and symbol not in state.universe:
        logger.warning("%s 不在股票池,拒绝", symbol)
        return

    # 2. 查当前价与前收价。
    current_price = get_current_price(symbol, state.current_date)
    if current_price is None:
        logger.warning("无法获取 %s 当前价", symbol)
        return
    prev_close = get_prev_close(symbol, state.current_date) or current_price

    # 3. 下单(Portfolio 内部做 T+1 / 整手 / 涨跌停校验)。
    record = state.portfolio.execute_trade(
        symbol=symbol,
        direction=direction,
        shares=amount,
        current_price=current_price,
        prev_close=prev_close,
        current_date=state.current_date,
    )

    # 4. 写审计日志 + 计数。
    entry: dict = {
        "date": state.current_date,
        "action": direction,
        "symbol": symbol,
        "shares": amount,
        "reasoning": reasoning,
        "filled": record is not None,
    }
    if record is not None:
        record.reasoning = reasoning
        entry["price"] = record.price
    state.audit_log.append(entry)

    state.actions_this_day += 1
    if state.actions_this_day >= state.max_actions_per_day:
        state.done = True

trade_agent_tools: list[Tool] = [
    get_price_data,
    get_macro_indicators,
    compute_alpha_factor,
    take_action,
]
