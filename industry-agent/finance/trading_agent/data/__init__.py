"""
Qlib 市场数据访问层：带超时保护的价格/指标查询封装 + 简单格式化 helper
"""

from __future__ import annotations

import pandas as pd

from .indicators import (
    BASE_FIELDS,
    EXPRESSION_PATTERN,
    INDICATOR_MAP,
    compute_rsi_python,
    parse_period_days,
    resolve_start_date,
)
from .qlib_client import (
    QLIB_TIMEOUT,
    ensure_qlib,
    fetch_weighted_close,
    get_current_price,
    get_prev_close,
    get_trading_calendar,
    qlib_dir,
    safe_features,
    update_portfolio_prices,
)

# --------------------------------------------------------------------- 格式化


def format_date(idx) -> str:
    """日期 → ``'MM-DD'``。接受 pd.Timestamp 或 str。"""
    if isinstance(idx, pd.Timestamp):
        return idx.strftime("%m-%d")
    return str(idx)[-5:]


def format_volume(vol: float) -> str:
    """成交量 → ``'1.2亿' / '34万' / '567'``。"""
    if vol >= 1e8:
        return f"{vol / 1e8:.1f}亿"
    if vol >= 1e4:
        return f"{vol / 1e4:.0f}万"
    return f"{vol:.0f}"


__all__ = [
    "BASE_FIELDS",
    "EXPRESSION_PATTERN",
    "INDICATOR_MAP",
    "QLIB_TIMEOUT",
    "compute_rsi_python",
    "ensure_qlib",
    "fetch_weighted_close",
    "format_date",
    "format_volume",
    "get_current_price",
    "get_prev_close",
    "get_trading_calendar",
    "parse_period_days",
    "qlib_dir",
    "resolve_start_date",
    "safe_features",
    "update_portfolio_prices",
]
