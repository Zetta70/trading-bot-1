"""
Async single-ticker trading bot.

Each TradingBot instance monitors one stock and makes independent
buy/sell decisions based on:
  1. ±0.5% momentum threshold (configurable)
  2. SMA(20) trend filter — BUY only above SMA, SELL only below
  3. ML bullish score gate — BUY only when score ≥ 0.75
  4. Trailing stop — protect profits on open positions

Multiple bots run concurrently under a PortfolioManager.
"""

import asyncio
import collections
import csv
import logging
from datetime import datetime
from pathlib import Path

from config import KST
from ml_predictor import MLPredictor

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")

# ── CSV Files ────────────────────────────────────────────────────────
TRADE_LOG = "trade_log.csv"
TRADE_HEADER = [
    "timestamp", "ticker", "side", "price", "fill_price",
    "qty", "order_no", "equity", "ml_score",
]

EQUITY_LOG = "equity_log.csv"
EQUITY_HEADER = ["timestamp", "ticker", "price", "equity"]


def _init_csv(path: str, header: list[str]) -> None:
    if not Path(path).exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


# Ensure CSVs exist at import time
_init_csv(TRADE_LOG, TRADE_HEADER)
_init_csv(EQUITY_LOG, EQUITY_HEADER)


class TradingBot:
    """
    Async trading bot for a single ticker.

    Exposes .equity, .last_price, .position for the PortfolioManager
    to read. Call .stop() to gracefully shut down.
    """

    def __init__(
        self,
        client,
        ticker: str,
        predictor: MLPredictor,
        *,
        qty: int = 1,
        threshold: float = 0.005,
        poll_interval: int = 10,
        sma_period: int = 20,
        bullish_threshold: float = 0.75,
        trailing_stop_pct: float = 0.02,
        initial_cash: int = 3_000_000,
    ):
        self.client = client
        self.ticker = ticker
        self.predictor = predictor
        self.qty = qty
        self.threshold = threshold
        self.poll_interval = poll_interval
        self.sma_period = sma_period
        self.bullish_threshold = bullish_threshold
        self.trailing_stop_pct = trailing_stop_pct

        # Price tracking
        self.reference_price: int | None = None
        self.last_price: int = 0
        self._price_history: list[int] = []
        self._sma_buf: collections.deque[int] = collections.deque(
            maxlen=sma_period,
        )

        # Portfolio per bot
        self.cash: int = initial_cash
        self.position: int = 0
        self._highest_since_entry: int = 0

        self._running = False

    # ── Properties ───────────────────────────────────────────────────

    @property
    def equity(self) -> int:
        """Current equity: cash + mark-to-market position value."""
        return self.cash + self.position * self.last_price

    def _now_str(self) -> str:
        return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

    # ── SMA ──────────────────────────────────────────────────────────

    def _get_sma(self) -> float | None:
        if len(self._sma_buf) < self.sma_period:
            return None
        return sum(self._sma_buf) / len(self._sma_buf)

    # ── Signal Evaluation ────────────────────────────────────────────

    def _evaluate(self, price: int) -> str | None:
        """
        Evaluate current price against strategy rules.

        Returns "BUY", "SELL", or None.
        """
        if self.reference_price is None:
            return None

        pct = (price - self.reference_price) / self.reference_price
        sma = self._get_sma()
        ml_score = self.predictor.score(self._price_history)

        # ── Trailing stop for open long positions ────────────────
        if self.position > 0:
            self._highest_since_entry = max(
                self._highest_since_entry, price,
            )
            trail_price = int(
                self._highest_since_entry * (1 - self.trailing_stop_pct)
            )
            if price <= trail_price:
                logger.info(
                    "[%s] TRAILING STOP: %d ≤ %d (peak=%d, trail=%.1f%%)",
                    self.ticker, price, trail_price,
                    self._highest_since_entry,
                    self.trailing_stop_pct * 100,
                )
                return "SELL"

        # ── BUY signal ───────────────────────────────────────────
        if pct >= self.threshold:
            if sma is not None and price <= sma:
                logger.debug(
                    "[%s] BUY suppressed: price %d ≤ SMA %.0f",
                    self.ticker, price, sma,
                )
                return None
            if ml_score < self.bullish_threshold:
                logger.info(
                    "[%s] BUY suppressed: ML %.3f < %.3f",
                    self.ticker, ml_score, self.bullish_threshold,
                )
                return None
            sma_str = f"{sma:,.0f}" if sma else "warmup"
            logger.info(
                "[%s] BUY SIGNAL: +%.2f%%, SMA=%s, ML=%.3f",
                self.ticker, pct * 100, sma_str, ml_score,
            )
            return "BUY"

        # ── SELL signal ──────────────────────────────────────────
        if pct <= -self.threshold:
            if sma is not None and price >= sma:
                logger.debug(
                    "[%s] SELL suppressed: price %d ≥ SMA %.0f",
                    self.ticker, price, sma,
                )
                return None
            logger.info(
                "[%s] SELL SIGNAL: %.2f%%",
                self.ticker, pct * 100,
            )
            return "SELL"

        return None

    # ── Order Execution ──────────────────────────────────────────────

    async def _execute(self, side: str, price: int) -> None:
        ml_score = self.predictor.score(self._price_history)
        try:
            result = await self.client.place_order(
                ticker=self.ticker, qty=self.qty, side=side,
                order_type="market",
            )
            output = result.get("output", {})
            order_no = output.get("ODNO", "N/A")
            fill_price = int(output.get("fill_price", str(price)))

            # Update portfolio
            if side == "BUY":
                self.cash -= fill_price * self.qty
                self.position += self.qty
                self._highest_since_entry = price
            else:
                self.cash += fill_price * self.qty
                self.position -= self.qty
                if self.position <= 0:
                    self._highest_since_entry = 0

            self.reference_price = fill_price

            # Log trade
            self._append_trade_csv(
                side, price, fill_price, order_no, ml_score,
            )
            trade_logger.info(
                "%s %s x%d @ %d (fill=%d) pos=%d equity=%d ML=%.3f",
                side, self.ticker, self.qty, price, fill_price,
                self.position, self.equity, ml_score,
            )

        except Exception as e:
            logger.error("[%s] Order failed: %s", self.ticker, e)

    # ── CSV Logging ──────────────────────────────────────────────────

    def _append_trade_csv(
        self, side: str, price: int, fill_price: int,
        order_no: str, ml_score: float,
    ) -> None:
        row = [
            self._now_str(), self.ticker, side, price, fill_price,
            self.qty, order_no, self.equity, f"{ml_score:.3f}",
        ]
        with open(TRADE_LOG, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _append_equity_csv(self, price: int) -> None:
        row = [self._now_str(), self.ticker, price, self.equity]
        with open(EQUITY_LOG, "a", newline="") as f:
            csv.writer(f).writerow(row)

    # ── Main Loop ────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Async main loop: poll → evaluate → execute → log → sleep.

        Runs until .stop() is called or the task is cancelled.
        """
        self._running = True
        logger.info(
            "[%s] Bot started (qty=%d, threshold=%.2f%%, "
            "sma=%d, trail=%.1f%%, cash=%s)",
            self.ticker, self.qty, self.threshold * 100,
            self.sma_period, self.trailing_stop_pct * 100,
            f"{self.cash:,}",
        )

        # Initial price
        try:
            price = await self.client.get_current_price(self.ticker)
            self.reference_price = price
            self.last_price = price
            self._price_history.append(price)
            self._sma_buf.append(price)
            logger.info("[%s] Initial price: %d KRW", self.ticker, price)
        except Exception as e:
            logger.error("[%s] Failed to start: %s", self.ticker, e)
            return

        while self._running:
            try:
                price = await self.client.get_current_price(self.ticker)
                self.last_price = price
                self._price_history.append(price)
                self._sma_buf.append(price)

                signal = self._evaluate(price)
                if signal:
                    await self._execute(signal, price)

                self._append_equity_csv(price)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[%s] Poll error: %s", self.ticker, e)

            try:
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break

        logger.info("[%s] Bot stopped.", self.ticker)

    def stop(self) -> None:
        """Signal the bot to exit after the current cycle."""
        self._running = False

    # ── State Persistence ────────────────────────────────────────────

    def get_state(self) -> dict:
        """Snapshot for JSON serialization."""
        return {
            "cash": self.cash,
            "position": self.position,
            "reference_price": self.reference_price,
            "last_price": self.last_price,
            "highest_since_entry": self._highest_since_entry,
        }

    def load_state(self, state: dict) -> None:
        """Restore from a saved snapshot."""
        self.cash = state.get("cash", self.cash)
        self.position = state.get("position", self.position)
        self.reference_price = state.get("reference_price")
        self.last_price = state.get("last_price", 0)
        self._highest_since_entry = state.get("highest_since_entry", 0)
