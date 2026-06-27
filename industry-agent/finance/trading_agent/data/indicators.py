"""技术指标名 → Qlib 表达式映射、period 解析、RSI Python fallback。"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Public name → Qlib expression. The agent's tool layer translates indicator
# names from this dict; anything not listed must go through compute_alpha_factor
# with a raw expression.
INDICATOR_MAP: dict[str, str] = {
    # Moving averages
    "MA5": "Mean($close, 5)",
    "MA10": "Mean($close, 10)",
    "MA20": "Mean($close, 20)",
    "MA60": "Mean($close, 60)",
    "EMA12": "EMA($close, 12)",
    "EMA26": "EMA($close, 26)",
    # Volatility
    "VOL_20": "Std($close / Ref($close, 1) - 1, 20)",
    # Momentum
    "ROC_5": "$close / Ref($close, 5) - 1",
    "ROC_20": "$close / Ref($close, 20) - 1",
    # Volume / price
    "VWAP": "Sum($close * $volume, 5) / Sum($volume, 5)",
    "VOL_RATIO_5_20": "Mean($volume, 5) / Mean($volume, 20)",
    # MACD (approximation)
    "MACD_DIF": "EMA($close, 12) - EMA($close, 26)",
    # Bollinger bands
    "BOLL_UPPER": "Mean($close, 20) + 2 * Std($close, 20)",
    "BOLL_LOWER": "Mean($close, 20) - 2 * Std($close, 20)",
    # RSI — Qlib expression has divide-by-zero risk; on exception fall back to
    # compute_rsi_python.
    "RSI_14": (
        "100 - 100 / (1 + "
        "Mean(If($close - Ref($close, 1) > 0, $close - Ref($close, 1), 0), 14) / "
        "Mean(If(Ref($close, 1) - $close > 0, Ref($close, 1) - $close, 0), 14))"
    ),
}

BASE_FIELDS: list[str] = ["$open", "$high", "$low", "$close", "$volume"]

EXPRESSION_PATTERN = re.compile(r"^[\w\s\$\+\-\*/\(\),\.\<\>\=\!&\|]+$")
_PERIOD_PATTERN = re.compile(r"^(\d+)d$")


def parse_period_days(period: str) -> int | None:
    """``'5d'`` → 5。格式非法返回 None。"""
    m = _PERIOD_PATTERN.match(period.strip())
    return int(m.group(1)) if m else None


def resolve_start_date(current_date: str, period_days: int) -> str:
    """从 ``current_date`` 往前推 ``period_days`` 天,加 1.5× buffer 覆盖非交易日。"""
    dt = datetime.strptime(current_date, "%Y-%m-%d")
    buffer_days = max(int(period_days * 1.5), period_days + 10)
    return (dt - timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def compute_rsi_python(close_series: pd.Series, period: int = 14) -> pd.Series:
    """SMA 版本的 RSI fallback,在 Qlib 表达式返回 inf/nan 时使用。"""
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rsi = pd.Series(np.nan, index=close_series.index)
    valid = pd.notna(avg_gain) & pd.notna(avg_loss)  # type: ignore[arg-type]
    both_zero = valid & (avg_gain == 0) & (avg_loss == 0)
    loss_zero = valid & (avg_loss == 0) & (avg_gain > 0)
    gain_zero = valid & (avg_gain == 0) & (avg_loss > 0)
    normal = valid & (avg_loss > 0) & (avg_gain >= 0)

    rsi[both_zero] = 50.0
    rsi[loss_zero] = 100.0
    rsi[gain_zero] = 0.0
    rs = avg_gain[normal] / avg_loss[normal]
    rsi[normal] = 100 - 100 / (1 + rs)
    return rsi
