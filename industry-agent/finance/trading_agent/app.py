"""Streamlit 面板：交互式运行 A 股交易 Agent。

启动（在本目录 trading_agent/ 下执行）：
    streamlit run app.py

依赖 Qlib 中国市场日频数据 + 一个 OpenAI 兼容推理端点（vLLM / OpenAI），
二者均在左侧边栏配置。运行前请先准备好 Qlib 数据，详见 README。
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
from datetime import date
from typing import Literal, cast

# 让 agent / data / backtest / schemas 在任意 CWD 下都能解析:把项目根目录(本文件所在目录)加进 sys.path。
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from agent import build_trade_agent  # noqa: E402
from agent.prompts import DEFAULT_SYSTEM_PROMPT  # noqa: E402
from backtest import (  # noqa: E402
    run_backtest_episode,
    run_multi_day_backtest,
)
from data import ensure_qlib, get_trading_calendar  # noqa: E402
from schemas import EnvConfig, Scenario  # noqa: E402

st.set_page_config(page_title="A 股交易 Agent", page_icon="📈", layout="wide")

# 说明:这些组件用 width="stretch"(streamlit 1.58 推荐写法,替代已废弃的
# use_container_width)。随附的类型 stub 仍把 width 标成 int|None(stub 滞后于运行时),
# 故各处加 `# type: ignore[arg-type]`。注意必须直接用 st.*,不能包一层别名——否则
# streamlit 的 magic 会把别名调用当成裸表达式自动 st.write,渲染出 DeltaGenerator 杂质。


# --------------------------------------------------------------------- 工具函数


def parse_universe(text: str) -> list[str]:
    """把逗号 / 空格 / 换行分隔的代码串解析成股票池列表。"""
    raw = text.replace(",", " ").replace("，", " ").split()
    return [s.strip().upper() for s in raw if s.strip()]


def parse_positions(df: pd.DataFrame) -> list[dict]:
    """把初始持仓表格转成 ``[{symbol, shares, price}, ...]``,跳过非法行。"""
    out: list[dict] = []
    for _, row in df.iterrows():
        symbol = str(row.get("代码", "") or "").strip().upper()
        if not symbol:
            continue
        try:
            shares = int(row["股数"])
            price = float(row["成本价"])
        except (ValueError, TypeError):
            continue
        if shares <= 0 or price <= 0:
            continue
        out.append({"symbol": symbol, "shares": shares, "price": price})
    return out


@st.cache_data(show_spinner=False)
def load_calendar(qlib_data_dir: str) -> list[str]:
    """加载 Qlib 交易日历(供多日回测枚举交易日)。"""
    ensure_qlib(qlib_data_dir)
    return get_trading_calendar()


def positions_to_df(positions: dict) -> pd.DataFrame:
    """把 ``dict[str, Position]`` 渲染成可读表格。"""
    rows = [
        {
            "代码": p.symbol,
            "股数": p.shares,
            "成本": round(p.cost_basis, 3),
            "现价": round(p.current_price, 3),
            "市值": round(p.market_value, 2),
            "浮盈": round(p.unrealized_pnl, 2),
            "浮盈%": f"{p.unrealized_pnl_pct:+.2%}",
        }
        for p in positions.values()
    ]
    return pd.DataFrame(rows)


def trades_to_df(trades: list) -> pd.DataFrame:
    """成交记录(TradeRecord 列表)→ DataFrame。"""
    return pd.DataFrame([dataclasses.asdict(t) for t in trades])


# --------------------------------------------------------------------- 侧边栏

with st.sidebar:
    st.header("⚙️ 配置")

    st.subheader("推理端点")
    base_url = st.text_input("base_url", value="http://localhost:8000/v1")
    api_key = st.text_input("api_key", value="EMPTY", type="password")
    model = st.text_input("模型名", value="Qwen3")

    st.subheader("数据 / Agent")
    qlib_data_dir = st.text_input("Qlib 数据目录", value="data/qlib_data")
    enable_thinking = st.checkbox("启用 reasoning（推理模型需勾选）", value=True)
    reasoning_effort = cast(
        Literal["low", "medium", "high"],
        st.selectbox("reasoning effort", ["low", "medium", "high"], index=1) or "medium",
    )
    max_turns = st.slider("单日最大对话轮数", min_value=5, max_value=40, value=20)

    with st.expander("查看系统提示词"):
        st.code(DEFAULT_SYSTEM_PROMPT, language="text")


# --------------------------------------------------------------------- 主面板

st.title("📈 A 股交易 Agent")
st.caption("LLM 自主查询行情、计算因子并决策买卖，在带撮合/滑点/手续费的回测环境中执行。")

mode = st.radio("运行模式", ["单日决策", "多日回测"], horizontal=True)

col_l, col_r = st.columns(2)
with col_l:
    universe_text = st.text_area(
        "股票池（逗号或空格分隔）",
        value="SH600519, SZ000001, SH601318",
        height=80,
    )
    initial_cash = st.number_input(
        "初始资金（元）", min_value=10_000.0, value=1_000_000.0, step=100_000.0
    )
with col_r:
    if mode == "单日决策":
        picked = st.date_input("交易日")
        start_date = picked.isoformat() if isinstance(picked, date) else ""
        end_date = start_date
    else:
        date_range = st.date_input("回测区间", value=())
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = date_range[0].isoformat(), date_range[1].isoformat()
        else:
            start_date = end_date = ""
    market_state_text = st.text_area("市场状态描述（可选，注入 prompt）", value="", height=80)

with st.expander("初始持仓（可选）"):
    pos_editor = st.data_editor(  # type: ignore[arg-type]
        pd.DataFrame({"代码": [], "股数": [], "成本价": []}),
        num_rows="dynamic",
        width="stretch",
        key="pos_editor",
    )

run = st.button("🚀 运行", type="primary", width="stretch")  # type: ignore[arg-type]


# --------------------------------------------------------------------- 执行


async def _run_single(scenario: Scenario, env_config: EnvConfig) -> object:
    agent = build_trade_agent(
        enable_thinking=enable_thinking, reasoning_effort=reasoning_effort
    )
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return await run_backtest_episode(
        scenario, agent, client, env_config=env_config, model=model, max_turns=max_turns
    )


async def _run_multi(scenarios: list[Scenario], env_config: EnvConfig) -> object:
    agent = build_trade_agent(
        enable_thinking=enable_thinking, reasoning_effort=reasoning_effort
    )
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return await run_multi_day_backtest(
        scenarios, agent, client, env_config=env_config, model=model,
        max_turns_per_day=max_turns,
    )


def render_single(result, scenario: Scenario) -> None:
    nav = result.portfolio.get_nav()
    ret = nav / scenario.initial_cash - 1 if scenario.initial_cash else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("期末总资产", f"{nav:,.0f}")
    c2.metric("累计收益", f"{ret:+.2%}")
    c3.metric("已实现盈亏", f"{result.portfolio.realized_pnl:+,.0f}")
    c4.metric("决策轮数", result.num_turns)
    c5.metric("成交笔数", len(result.trades))

    st.subheader("🤖 Agent 决策日志")
    if result.audit_log:
        st.dataframe(pd.DataFrame(result.audit_log), width="stretch")  # type: ignore[arg-type]
    else:
        st.info("Agent 未产生有效决策（可能是端点不可用或当日无数据）。")

    st.subheader("📦 收盘持仓（清盘前快照）")
    pos_df = positions_to_df(result.final_positions)
    if not pos_df.empty:
        st.dataframe(pos_df, width="stretch")  # type: ignore[arg-type]
    else:
        st.write("空仓")

    st.subheader("🧾 成交明细")
    tdf = trades_to_df(result.trades)
    if not tdf.empty:
        st.dataframe(tdf, width="stretch")  # type: ignore[arg-type]
    else:
        st.write("无成交")


def render_multi(result) -> None:
    nav0 = result.nav_curve[0] if result.nav_curve else 0.0
    navN = result.nav_curve[-1] if result.nav_curve else 0.0
    ret = navN / nav0 - 1 if nav0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("期末总资产", f"{navN:,.0f}")
    c2.metric("区间累计收益", f"{ret:+.2%}")
    c3.metric("交易日数", len(result.daily_results))

    st.subheader("📈 净值曲线")
    st.line_chart(pd.DataFrame({"NAV": result.nav_curve}))

    st.subheader("📅 每日明细")
    daily = pd.DataFrame(
        [
            {
                "日期": d.date,
                "日初NAV": round(d.nav_before, 2),
                "日末NAV": round(d.nav_after, 2),
                "日收益": f"{d.daily_return:+.2%}",
                "成交笔数": len(d.trades_this_day),
                "决策轮数": d.num_turns,
            }
            for d in result.daily_results
        ]
    )
    st.dataframe(daily, width="stretch")  # type: ignore[arg-type]


if run:
    universe = parse_universe(universe_text)
    initial_positions = parse_positions(pos_editor)
    env_config = EnvConfig(qlib_data_dir=qlib_data_dir, initial_cash=initial_cash)

    if not universe:
        st.error("股票池为空，请至少填入一个代码（如 SH600519）。")
    elif not start_date:
        st.error("请选择交易日 / 回测区间。")
    else:
        try:
            with st.spinner("Agent 运行中……（首次调用会初始化 Qlib，可能稍慢）"):
                if mode == "单日决策":
                    scenario = Scenario(
                        start_date=start_date,
                        universe=universe,
                        initial_cash=initial_cash,
                        initial_positions=initial_positions,
                        market_state_text=market_state_text,
                    )
                    result = asyncio.run(_run_single(scenario, env_config))
                    render_single(result, scenario)
                else:
                    calendar = load_calendar(qlib_data_dir)
                    dates = [d for d in calendar if start_date <= d <= end_date]
                    if not dates:
                        st.error("所选区间内无交易日，请检查日期或 Qlib 数据。")
                        st.stop()
                    if len(dates) > 20:
                        st.warning(f"区间含 {len(dates)} 个交易日，运行时间较长。")
                    scenarios = [
                        Scenario(
                            start_date=d,
                            universe=universe,
                            initial_cash=initial_cash,
                            initial_positions=initial_positions,
                            market_state_text=market_state_text,
                        )
                        for d in dates
                    ]
                    result = asyncio.run(_run_multi(scenarios, env_config))
                    render_multi(result)
            st.success("完成 ✅")
        except Exception as e:  # noqa: BLE001 — 面板兜底,把异常显示给用户
            st.error(f"运行失败：{type(e).__name__}: {e}")
            st.exception(e)
