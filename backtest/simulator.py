"""
Trading Simulator — realistic execution engine for backtesting.

Models:
  - Commission (buy + sell)
  - Securities transaction tax (sell-side, KOSPI)
  - Slippage (both sides)
  - Position sizing with max position % and max total positions
  - Stop-loss triggered by intraday low (not just close)
  - Next-day open execution (no look-ahead)

Trade and equity logs match the live bot's CSV format.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a completed (round-trip) trade."""
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    return_pct: float
    holding_days: int
    exit_reason: str  # 'signal' or 'stop_loss'


@dataclass
class Position:
    """Open position tracker."""
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    peak_price: float  # for trailing / stop-loss


class TradingSimulator:
    """
    Event-driven trading simulator.

    Signals are generated after market close. Execution happens at the
    **next trading day's open price**, with costs applied.

    Parameters
    ----------
    initial_capital : int
        Starting cash in KRW.
    commission : float
        One-way commission rate (applied to both buy and sell).
    tax : float
        Securities transaction tax (sell-side only, KOSPI 0.18%).
    slippage : float
        One-way slippage rate.
    max_position_pct : float
        Maximum fraction of portfolio allocated to a single position.
    max_positions : int
        Maximum number of concurrent open positions.
    stop_loss_pct : float
        Stop-loss threshold (negative, e.g. -0.07 = -7%).
    threshold_buy : float
        Minimum predicted probability to trigger a BUY.
    threshold_sell : float
        Predicted probability below which to trigger a SELL.
    """

    def __init__(
        self,
        initial_capital: int = 10_000_000,
        commission: float = 0.00015,
        tax: float = 0.0018,
        slippage: float = 0.001,
        max_position_pct: float = 0.10,
        max_positions: int = 10,
        stop_loss_pct: float = -0.07,
        threshold_buy: float = 0.55,
        threshold_sell: float = 0.45,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.tax = tax
        self.slippage = slippage
        self.max_position_pct = max_position_pct
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell

        # State
        self.cash: float = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.daily_equity: list[dict] = []

    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.daily_equity.clear()

    # ── Cost Calculations ────────────────────────────────────────

    def _buy_cost(self, price: float, shares: int) -> float:
        """Total cost to open a long position."""
        return price * shares * (1 + self.commission + self.slippage)

    def _sell_proceeds(self, price: float, shares: int) -> float:
        """Net proceeds from closing a long position."""
        return price * shares * (1 - self.commission - self.tax - self.slippage)

    # ── Execution ────────────────────────────────────────────────

    def _execute_buy(
        self,
        ticker: str,
        date: pd.Timestamp,
        open_price: float,
    ) -> bool:
        """
        Execute a buy order at the given open price.

        Returns True if executed, False if skipped.
        """
        if ticker in self.positions:
            return False
        if len(self.positions) >= self.max_positions:
            return False

        # Position sizing: max_position_pct of current portfolio value
        portfolio_value = self._portfolio_value_at(open_price)
        max_alloc = portfolio_value * self.max_position_pct
        cost_per_share = open_price * (1 + self.commission + self.slippage)

        if cost_per_share <= 0:
            return False

        shares = int(max_alloc / cost_per_share)
        if shares <= 0:
            return False

        total_cost = self._buy_cost(open_price, shares)
        if total_cost > self.cash:
            shares = int(self.cash / cost_per_share)
            if shares <= 0:
                return False
            total_cost = self._buy_cost(open_price, shares)

        self.cash -= total_cost
        self.positions[ticker] = Position(
            ticker=ticker,
            entry_date=date,
            entry_price=open_price,
            shares=shares,
            peak_price=open_price,
        )
        return True

    def _execute_sell(
        self,
        ticker: str,
        date: pd.Timestamp,
        price: float,
        reason: str = "signal",
    ) -> bool:
        """
        Close a position at the given price.

        Returns True if executed, False if no position exists.
        """
        if ticker not in self.positions:
            return False

        pos = self.positions.pop(ticker)
        proceeds = self._sell_proceeds(price, pos.shares)
        self.cash += proceeds

        cost_basis = self._buy_cost(pos.entry_price, pos.shares)
        pnl = proceeds - cost_basis
        ret = pnl / cost_basis if cost_basis > 0 else 0.0
        holding = (date - pos.entry_date).days

        self.trades.append(Trade(
            ticker=ticker,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=price,
            shares=pos.shares,
            pnl=pnl,
            return_pct=ret,
            holding_days=max(holding, 1),
            exit_reason=reason,
        ))
        return True

    def _portfolio_value_at(self, fallback_price: float = 0) -> float:
        """Current portfolio value (approximate, uses entry price for open positions)."""
        pos_value = sum(
            p.entry_price * p.shares for p in self.positions.values()
        )
        return self.cash + pos_value

    # ── Daily Step ───────────────────────────────────────────────

    def step(
        self,
        date: pd.Timestamp,
        ticker: str,
        ohlc: dict,
        signal_prob: float | None,
    ) -> None:
        """
        Process one day for one ticker.

        Parameters
        ----------
        date : pd.Timestamp
            Current trading date.
        ticker : str
            Stock ticker.
        ohlc : dict
            Keys: 'open', 'high', 'low', 'close' for this date.
        signal_prob : float or None
            Predicted probability from the **previous day's** model.
            None means no signal (e.g. first day).
        """
        open_price = ohlc["open"]
        low_price = ohlc["low"]
        close_price = ohlc["close"]

        # ── 1. Check stop-loss on open positions (intraday low) ──
        if ticker in self.positions:
            pos = self.positions[ticker]
            pos.peak_price = max(pos.peak_price, ohlc["high"])
            loss_from_entry = (low_price - pos.entry_price) / pos.entry_price

            if loss_from_entry <= self.stop_loss_pct:
                # Stop-loss triggered — use the stop-loss price, not low
                stop_price = pos.entry_price * (1 + self.stop_loss_pct)
                self._execute_sell(ticker, date, stop_price, reason="stop_loss")

        # ── 2. Execute signals from previous day's close ─────────
        if signal_prob is not None:
            if signal_prob >= self.threshold_buy:
                self._execute_buy(ticker, date, open_price)
            elif signal_prob <= self.threshold_sell:
                self._execute_sell(ticker, date, open_price, reason="signal")

        # ── 3. Record daily equity ───────────────────────────────
        mark_to_market = sum(
            close_price * p.shares
            if p.ticker == ticker
            else p.entry_price * p.shares
            for p in self.positions.values()
        )
        total_equity = self.cash + mark_to_market

        self.daily_equity.append({
            "timestamp": date,
            "ticker": ticker,
            "price": close_price,
            "equity": total_equity,
        })

    def step_portfolio(
        self,
        date: pd.Timestamp,
        ticker_data: dict[str, dict],
        signal_probs: dict[str, float | None],
        close_prices: dict[str, float],
    ) -> None:
        """
        Process one day for all tickers at once.

        Parameters
        ----------
        date : Current trading date.
        ticker_data : {ticker: {open, high, low, close}} for this date.
        signal_probs : {ticker: prob or None} from previous day.
        close_prices : {ticker: close_price} for mark-to-market.
        """
        # 1. Stop-loss checks first (across all tickers)
        for ticker in list(self.positions.keys()):
            if ticker not in ticker_data:
                continue
            ohlc = ticker_data[ticker]
            pos = self.positions[ticker]
            pos.peak_price = max(pos.peak_price, ohlc["high"])
            loss = (ohlc["low"] - pos.entry_price) / pos.entry_price
            if loss <= self.stop_loss_pct:
                stop_price = pos.entry_price * (1 + self.stop_loss_pct)
                self._execute_sell(ticker, date, stop_price, reason="stop_loss")

        # 2. Execute signals
        for ticker, prob in signal_probs.items():
            if prob is None or ticker not in ticker_data:
                continue
            open_price = ticker_data[ticker]["open"]
            if prob >= self.threshold_buy:
                self._execute_buy(ticker, date, open_price)
            elif prob <= self.threshold_sell:
                self._execute_sell(ticker, date, open_price, reason="signal")

        # 3. Mark-to-market and record equity
        mark_to_market = sum(
            close_prices.get(p.ticker, p.entry_price) * p.shares
            for p in self.positions.values()
        )
        total_equity = self.cash + mark_to_market

        self.daily_equity.append({
            "timestamp": date,
            "ticker": "PORTFOLIO",
            "price": 0,
            "equity": total_equity,
        })

    # ── Results ──────────────────────────────────────────────────

    def get_trade_log(self) -> pd.DataFrame:
        """
        Return completed trades as a DataFrame.

        Columns match the live bot's trade_log.csv structure.
        """
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "timestamp": t.exit_date,
                "ticker": t.ticker,
                "side": "SELL",
                "price": t.exit_price,
                "fill_price": t.exit_price,
                "qty": t.shares,
                "order_no": "BT",
                "equity": 0,
                "ml_score": 0,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "return_pct": t.return_pct,
                "holding_days": t.holding_days,
                "exit_reason": t.exit_reason,
            })
        return pd.DataFrame(records)

    def get_equity_log(self) -> pd.DataFrame:
        """
        Return daily equity as a DataFrame.

        Columns match the live bot's equity_log.csv structure:
        timestamp, ticker, price, equity.
        """
        if not self.daily_equity:
            return pd.DataFrame()
        return pd.DataFrame(self.daily_equity)
