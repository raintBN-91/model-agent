"""A 股账户模拟器:持仓跟踪、撮合、NAV 计算。"""

from __future__ import annotations

import logging

from data.qlib_client import get_trading_calendar
from schemas import PRICE_LIMIT_PCT, EnvConfig, Position, Scenario, TradeRecord

logger = logging.getLogger(__name__)


class Portfolio:
    """单账户撮合引擎。每次 ``execute_trade`` 都会校验整手/T+1/涨跌停,撮合参数对齐 Qlib ``exchange_kwargs``。"""

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        open_cost: float = 0.0005,
        close_cost: float = 0.0015,
        min_cost: float = 5.0,
        slippage_rate: float = 0.0005,
    ) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.open_cost = open_cost
        self.close_cost = close_cost
        self.min_cost = min_cost
        self.slippage_rate = slippage_rate

        self.positions: dict[str, Position] = {}
        self.trade_history: list[TradeRecord] = []
        self.realized_pnl: float = 0.0

    # ----------------------------------------------------------------- 工厂

    @classmethod
    def from_scenario(cls, scenario: Scenario, env_config: EnvConfig) -> Portfolio:
        """根据 Scenario + EnvConfig 构造 Portfolio 并塞入初始持仓。

        初始持仓的 ``buy_date`` 设为前一交易日(从 Qlib 日历查),这样 T+1 规则
        不会阻塞 day-0 卖出。日历查不到时回退到 ``start_date``,代价是 day-0
        不能卖,但仍可买。
        """
        portfolio = cls(
            initial_cash=scenario.initial_cash,
            open_cost=env_config.open_cost,
            close_cost=env_config.close_cost,
            min_cost=env_config.min_cost,
            slippage_rate=env_config.slippage_rate,
        )
        if not scenario.initial_positions:
            return portfolio

        # buy_date 设为前一交易日,这样 T+1 不卡 day-0 卖出。
        prev_date = scenario.start_date
        try:
            cal = get_trading_calendar()
            t_idx = cal.index(scenario.start_date)
            if t_idx > 0:
                prev_date = cal[t_idx - 1]
        except (ValueError, IndexError):
            logger.debug("日历查询失败 %s,使用 start_date 作为 prev_date", scenario.start_date)

        for p in scenario.initial_positions:
            portfolio.add_initial_position(
                symbol=p["symbol"],
                shares=int(p["shares"]),
                price=float(p["price"]),
                buy_date=prev_date,
            )
        return portfolio

    # ----------------------------------------------------------------- 初始化

    def add_initial_position(
        self, symbol: str, shares: int, price: float, buy_date: str,
    ) -> None:
        """直接添加初始持仓,不计手续费滑点。``buy_date`` 应填前一交易日以便 day-0 可卖。"""
        self.positions[symbol] = Position(
            symbol=symbol,
            shares=shares,
            cost_basis=price,
            current_price=price,
            buy_date=buy_date,
        )
        self.cash -= shares * price

    # ----------------------------------------------------------------- 撮合

    def execute_trade(
        self,
        symbol: str,
        direction: str,
        shares: int,
        current_price: float,
        prev_close: float,
        current_date: str,
    ) -> TradeRecord | None:
        """执行一笔买卖。返回 TradeRecord;校验失败返回 None。校验顺序:整手 → T+1 → 涨跌停 → 现金/持仓。"""
        if shares % 100 != 0:
            logger.debug("拒绝:%s %s %d 股非整手", symbol, direction, shares)
            return None

        if direction == "sell":
            pos = self.positions.get(symbol)
            if pos is None:
                logger.debug("拒绝:%s 无持仓", symbol)
                return None
            if pos.buy_date == current_date:
                logger.debug("拒绝:%s T+1 限制(当日买入)", symbol)
                return None

        if prev_close > 0:
            change_pct = abs(current_price / prev_close - 1)
            # 1e-9 容差防止恰好打板的浮点抖动。
            if change_pct > PRICE_LIMIT_PCT + 1e-9:
                logger.debug(
                    "拒绝:%s 涨跌停 (%.2f%% > %.0f%%)",
                    symbol, change_pct * 100, PRICE_LIMIT_PCT * 100,
                )
                return None

        if direction == "buy":
            return self._execute_buy(symbol, shares, current_price, current_date)
        if direction == "sell":
            return self._execute_sell(symbol, shares, current_price, current_date)

        logger.warning("未知方向:%s", direction)
        return None

    def _execute_buy(
        self, symbol: str, shares: int, base_price: float, current_date: str,
    ) -> TradeRecord | None:
        """买入撮合:滑点 → 手续费 → 现金校验 → 加权平均成本更新。现金不足返回 None。"""
        deal_price = base_price * (1 + self.slippage_rate)
        slippage_cost = (deal_price - base_price) * shares
        commission = max(deal_price * shares * self.open_cost, self.min_cost)
        total_cost = deal_price * shares + commission

        if total_cost > self.cash:
            logger.debug(
                "拒绝:%s 买 %d 股需 %.2f,现金 %.2f",
                symbol, shares, total_cost, self.cash,
            )
            return None

        self.cash -= total_cost

        if symbol in self.positions:
            pos = self.positions[symbol]
            total_shares = pos.shares + shares
            pos.cost_basis = (pos.cost_basis * pos.shares + deal_price * shares) / total_shares
            pos.shares = total_shares
            pos.current_price = base_price
            pos.buy_date = current_date
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                cost_basis=deal_price,
                current_price=base_price,
                buy_date=current_date,
            )

        record = TradeRecord(
            date=current_date,
            symbol=symbol,
            direction="buy",
            shares=shares,
            price=deal_price,
            commission=commission,
            slippage_cost=slippage_cost,
        )
        self.trade_history.append(record)
        logger.debug("买入 %s %d 股 @ %.4f 手续费=%.2f", symbol, shares, deal_price, commission)
        return record

    def _execute_sell(
        self, symbol: str, shares: int, base_price: float, current_date: str,
    ) -> TradeRecord | None:
        """卖出撮合:滑点 → 手续费 → 实现盈亏入账;持仓归零自动删除条目。持仓不足返回 None。"""
        pos = self.positions.get(symbol)
        if pos is None or pos.shares < shares:
            logger.debug(
                "拒绝:%s 卖 %d 股,持仓 %d",
                symbol, shares, pos.shares if pos else 0,
            )
            return None

        deal_price = base_price * (1 - self.slippage_rate)
        slippage_cost = (base_price - deal_price) * shares
        commission = max(deal_price * shares * self.close_cost, self.min_cost)
        proceeds = deal_price * shares - commission

        trade_pnl = (deal_price - pos.cost_basis) * shares
        self.realized_pnl += trade_pnl

        self.cash += proceeds
        pos.shares -= shares

        if pos.shares == 0:
            del self.positions[symbol]
        else:
            pos.current_price = base_price

        record = TradeRecord(
            date=current_date,
            symbol=symbol,
            direction="sell",
            shares=shares,
            price=deal_price,
            commission=commission,
            slippage_cost=slippage_cost,
        )
        self.trade_history.append(record)
        logger.debug("卖出 %s %d 股 @ %.4f 手续费=%.2f", symbol, shares, deal_price, commission)
        return record

    # ----------------------------------------------------------------- NAV / 收盘

    def update_prices(self, prices: dict[str, float]) -> None:
        """批量更新持仓现价。"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

    def get_nav(self) -> float:
        """总资产 = 现金 + 所有持仓市值。"""
        return self.cash + sum(pos.market_value for pos in self.positions.values())

    def liquidate_all(self, current_date: str) -> list[TradeRecord]:
        """强制清盘所有持仓(跳过 T+1 与整手校验,但计滑点手续费)。"""
        records: list[TradeRecord] = []
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            record = self._execute_sell(symbol, pos.shares, pos.current_price, current_date)
            if record:
                record.reasoning = "回测结束强制清盘"
                records.append(record)
        return records
