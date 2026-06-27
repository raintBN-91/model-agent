"""TakeAction 之后返回给 agent 的观测文本生成器。"""

from __future__ import annotations

from .portfolio import Portfolio
from data.qlib_client import get_current_price


def format_portfolio_stats(portfolio: Portfolio) -> str:
    """
    把持仓、累计收益、已实现盈亏渲染成 markdown 块。向 Agent 提供可读持仓信息，帮助其决策。
    """
    nav = portfolio.get_nav()
    cumulative_return = nav / portfolio.initial_cash - 1 if portfolio.initial_cash > 0 else 0.0
    lines: list[str] = [
        "## 当前持仓状态\n",
        f"总资产：{nav:.2f}  现金：{portfolio.cash:.2f}",
        f"累计收益：{cumulative_return:+.2%}  已实现盈亏：{portfolio.realized_pnl:+.2f}",
    ]

    if portfolio.positions:
        lines.append(f"\n持仓明细（{len(portfolio.positions)}只）：")
        for pos in portfolio.positions.values():
            weight = pos.market_value / nav if nav > 0 else 0.0
            lines.append(
                f"- {pos.symbol}: {pos.shares}股 "
                f"成本{pos.cost_basis:.2f} 现价{pos.current_price:.2f} "
                f"浮盈{pos.unrealized_pnl:+.2f}({pos.unrealized_pnl_pct:+.1%}) "
                f"占比{weight:.1%}"
            )
    else:
        lines.append("\n当前无持仓")

    return "\n".join(lines)


def format_observation(
    portfolio: Portfolio,
    current_date: str,
    universe: list[str],
    actions_this_day: int,
    max_actions_per_day: int,
    done: bool,
    *,
    max_position_pct: float = 0.40,
    lot_size: int = 100,
) -> str:
    """
    完整观测：持仓 + 股票池价格与可买上限 + 当日交易计数。

    ``done=True`` 时只返回"交易日结束。",避免 agent 继续生成无意义内容。
    """
    if done:
        return "交易日结束。"

    parts: list[str] = [format_portfolio_stats(portfolio)]

    if universe:
        cash = portfolio.cash
        nav = portfolio.get_nav()

        lines = ["## 股票池最新价格\n"]
        lines.append(f"可用现金：{cash:,.2f} 元（总资产 {nav:,.2f} 元）\n")
        for symbol in universe:
            price = get_current_price(symbol, current_date)
            if price is None:
                lines.append(f"- {symbol}: 价格不可用")
                continue

            deal_price = price * (1 + portfolio.slippage_rate)
            cost_per_share = deal_price * (1 + portfolio.open_cost)

            max_shares_by_cash = int(cash / cost_per_share) if cost_per_share > 0 else 0
            max_lots_by_cash = max_shares_by_cash // lot_size
            if max_lots_by_cash > 0:
                gross = deal_price * max_lots_by_cash * lot_size
                commission = max(gross * portfolio.open_cost, portfolio.min_cost)
                if gross + commission > cash:
                    max_lots_by_cash -= 1

            existing_value = (
                portfolio.positions[symbol].market_value if symbol in portfolio.positions else 0.0
            )
            remaining_budget = max(nav * max_position_pct - existing_value, 0.0)
            max_lots_by_limit = int(remaining_budget / deal_price) // lot_size if deal_price > 0 else 0

            max_lots = max(min(max_lots_by_cash, max_lots_by_limit), 0)
            max_shares = max_lots * lot_size
            lot_cost = price * lot_size

            constraint_hint = ""
            if max_lots_by_limit < max_lots_by_cash:
                pct = existing_value / nav * 100 if nav > 0 else 0
                constraint_hint = (
                    f"（当前持仓占比{pct:.0f}%，受{int(max_position_pct * 100)}%上限约束）"
                )

            lines.append(
                f"- {symbol}: 现价 {price:.2f}，{lot_size}股≈{lot_cost:,.0f}元，"
                f"最多可买 {max_shares} 股{constraint_hint}"
            )
        parts.extend(["", "\n".join(lines)])

    parts.append(f"\n（已执行 {actions_this_day}/{max_actions_per_day} 次交易）")
    return "\n".join(parts)
