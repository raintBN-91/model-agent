"""
Qlib 数据访问层与价格查询格式化层

底层：进程级 Qlib 单例 + 带超时的 ``D.features`` 封装。
Qlib 内部用 joblib worker,worker 崩了可能让父进程永远 block；``safe_features`` 走 daemon 线程 + 硬超时 join,坏请求干净失败而不是卡死 rollout。

业务级 helper:基于 ``safe_features`` 封装的价格查询，供 backtest 与 agent tools 使用。
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd
import qlib
from qlib.data import D

if TYPE_CHECKING:
    from backtest.portfolio import Portfolio

logger = logging.getLogger(__name__)


# ---- joblib >= 1.5 compat: qlib `ParallelExt._backend_args` 不存在 ----------
# qlib 0.9.6 的 `ParallelExt.__init__` 给 `self._backend_args["maxtasksperchild"]`
# 赋值,但 joblib 1.5+ 删了 `_backend_args` 属性,导致每次 D.features 调用都
# `AttributeError: 'ParallelExt' object has no attribute '_backend_args'`。
# 这只是个性能优化(限制子进程任务复用),skip 不影响结果,只影响 worker 复用。
def _patch_qlib_parallel_ext() -> None:
    try:
        from qlib.utils.paral import MultiprocessingBackend, ParallelExt
    except ImportError:
        return
    if getattr(ParallelExt, "_trade_agent_patched", False):
        return
    _orig_init = ParallelExt.__init__

    def _patched_init(self, *args, **kwargs):
        maxtasksperchild = kwargs.pop("maxtasksperchild", None)
        # 跳过 qlib 的 _backend_args 赋值,直接 Parallel.__init__
        Parallel = ParallelExt.__mro__[1]  # joblib.Parallel
        Parallel.__init__(self, *args, **kwargs)
        if isinstance(self._backend, MultiprocessingBackend):
            try:
                self._backend_args["maxtasksperchild"] = maxtasksperchild
            except AttributeError:
                pass  # joblib >= 1.5 removed _backend_args

    ParallelExt.__init__ = _patched_init
    ParallelExt._trade_agent_patched = True


_patch_qlib_parallel_ext()

QLIB_TIMEOUT = 30  # seconds — applies per D.features call.

_qlib_initialized: bool = False
_init_lock = threading.Lock()


def qlib_dir(config: Any) -> str | None:
    """从 tool config dict 中取 ``qlib_data_dir``,非 dict 返回 None。"""
    return config.get("qlib_data_dir") if isinstance(config, dict) else None


def ensure_qlib(qlib_data_dir: str | None = None) -> None:
    """幂等的 Qlib 初始化,支持多线程并发调用。"""
    global _qlib_initialized
    if _qlib_initialized:
        return
    with _init_lock:
        if _qlib_initialized:
            return
        if qlib_data_dir:
            try:
                qlib.init(provider_uri=qlib_data_dir, region="cn")
                logger.debug("Qlib initialized: %s", qlib_data_dir)
            except Exception:
                logger.debug("Qlib already initialized, skipping")
        _qlib_initialized = True


def get_trading_calendar(freq: str = "day") -> list[str]:
    """返回 Qlib 日历的全部交易日 ``YYYY-MM-DD`` 字符串列表。"""
    return [d.strftime("%Y-%m-%d") for d in D.calendar(freq=freq)]


def safe_features(
    instruments: list[str],
    fields: list[str],
    start_time: str,
    end_time: str,
    freq: str = "day",
) -> pd.DataFrame:
    """
    带超时的 ``qlib.data.D.features`` 封装。

    超过 ``QLIB_TIMEOUT`` 秒抛 ``concurrent.futures.TimeoutError``;
    调用方应捕获并返回错误字符串,不要让 agent loop 崩溃。

    此处实现是在训练过程中,我们发现在 worker 进程中调用 Qlib 的 D.features 时,偶尔会遇到
    Qlib 内部的 joblib worker 崩溃导致父进程永远 block 的情况。为了避免这种情况,我们使用
    了一个 daemon 线程来调用 D.features,并设置了一个硬超时 join。如果在指定时间内线程仍然在
    运行,我们就认为它已经卡死了,并抛出一个 TimeoutError。这样可以让坏请求干净失败,而不是
    卡死整个 step。
    """
    result: list[pd.DataFrame | None] = [None]
    error: list[BaseException | None] = [None]

    def _call() -> None:
        try:
            result[0] = D.features(
                instruments, fields,
                start_time=start_time, end_time=end_time, freq=freq,
            )
        except BaseException as e:  # noqa: BLE001 — propagate via list, raise after join
            error[0] = e

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=QLIB_TIMEOUT)

    if t.is_alive():
        logger.error("D.features timed out (%ss), instruments=%s", QLIB_TIMEOUT, instruments)
        raise concurrent.futures.TimeoutError(f"D.features timed out after {QLIB_TIMEOUT}s")

    if error[0] is not None:
        raise error[0]

    assert result[0] is not None
    return result[0]


def get_current_price(symbol: str, current_date: str) -> float | None:
    """当日收盘价;数据缺失或超时返回 None。"""
    try:
        df = safe_features([symbol], ["$close"], start_time=current_date, end_time=current_date)
        if df.empty:
            return None
        return float(df.iloc[-1, 0])
    except concurrent.futures.TimeoutError:
        logger.error("[get_current_price] Qlib timeout (%ss), symbol=%s", QLIB_TIMEOUT, symbol)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Fetch %s price failed: %s", symbol, e)
        return None


def get_prev_close(symbol: str, current_date: str) -> float | None:
    """前一交易日收盘价,涨跌停校验用。"""
    try:
        # 用 datetime 而非 pd.Timestamp,避免 NaT 类型分支。
        start_dt = datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=10)
        df = safe_features(
            [symbol], ["$close"],
            start_time=start_dt.strftime("%Y-%m-%d"),
            end_time=current_date,
        )
        if df.empty or len(df) < 2:
            return None
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return float(df.iloc[-2, 0])
    except concurrent.futures.TimeoutError:
        logger.error("[get_prev_close] Qlib timeout (%ss), symbol=%s", QLIB_TIMEOUT, symbol)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Fetch %s prev close failed: %s", symbol, e)
        return None


def update_portfolio_prices(portfolio: Portfolio, current_date: str) -> None:
    """批量刷新每只持仓的 ``Position.current_price``。"""
    prices: dict[str, float] = {}
    for symbol in portfolio.positions:
        price = get_current_price(symbol, current_date)
        if price is not None:
            prices[symbol] = price
    if prices:
        portfolio.update_prices(prices)


def fetch_weighted_close(
    symbol: str,
    future_dates: list[str],
    weights: list[float],
    total_w: float,
    fallback_price: float = 0.0,
) -> float:
    """前瞻加权收盘价;缺失日期自动归一化剩余权重。"""
    weighted_price = 0.0
    weight_sum = 0.0
    for i, fdate in enumerate(future_dates):
        try:
            df = safe_features([symbol], ["$close"], start_time=fdate, end_time=fdate)
            if df.empty:
                continue
            if isinstance(df.index, pd.MultiIndex) and symbol in df.index.get_level_values(0):
                close = float(df.xs(symbol, level=0)["$close"].iloc[-1])
            else:
                close = float(df.iloc[-1, 0])
            if close > 0:
                weighted_price += weights[i] * close
                weight_sum += weights[i]
        except concurrent.futures.TimeoutError:
            logger.error("[forward_reward] %s %s Qlib timeout", symbol, fdate)
        except Exception as e:  # noqa: BLE001
            logger.debug("[forward_reward] %s %s price fetch failed: %s", symbol, fdate, e)

    if weight_sum < 1e-9:
        return fallback_price
    if weight_sum < total_w - 1e-9:
        return weighted_price / weight_sum
    return weighted_price
