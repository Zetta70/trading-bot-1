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
import time
import warnings
from datetime import datetime
from pathlib import Path

from config import KST
from ml_predictor import MLPredictor
from ohlcv_cache import get_cache

# Sentinel for the deprecated `qty` __init__ kwarg (Patch 2). Lets us
# distinguish "user supplied qty=N" from "default applied", so the
# deprecation warning fires only on actual use.
_QTY_UNSET = object()


class _NullAsyncLock:
    """No-op async lock for TradingBots constructed without a portfolio
    (Patch 5). Lets ``async with bot._cash_lock_ctx():`` work uniformly
    whether a real lock is plumbed through or not."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

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
        qty=_QTY_UNSET,  # DEPRECATED (Patch 2) — qty is now sized dynamically
        threshold: float = 0.005,
        poll_interval: int = 10,
        sma_period: int = 20,
        bullish_threshold: float = 0.75,
        trailing_stop_pct: float = 0.02,
        initial_cash: int = 3_000_000,
        commission: float = 0.00015,
        tax_kr: float = 0.0023,
        tax_us: float = 0.0,
        entry_mode: str = "eod",
        max_trades_per_day: int = 2,
        currency: str = "KRW",
        cash_lock: asyncio.Lock | None = None,
    ):
        self.client = client
        self.ticker = ticker
        self.predictor = predictor
        # qty is deprecated (Patch 2). Keep the attribute for any external
        # readers, but _execute now sizes BUY by cash and SELL by position.
        if qty is _QTY_UNSET:
            self.qty = 1  # legacy default for any read-only consumers
        else:
            warnings.warn(
                "TradingBot(qty=...) is deprecated; qty is now sized "
                "dynamically inside _execute (BUY: cash * 0.95 / "
                "cost-per-share; SELL: full liquidation). The argument "
                "is accepted for backward compatibility but ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.qty = qty
        self.threshold = threshold
        self.poll_interval = poll_interval
        self.sma_period = sma_period
        self.bullish_threshold = bullish_threshold
        self.trailing_stop_pct = trailing_stop_pct
        self.commission = commission
        self.tax_kr = tax_kr
        self.tax_us = tax_us
        self.currency = currency.upper()
        if self.currency not in ("KRW", "USD"):
            raise ValueError(
                f"currency must be 'KRW' or 'USD', got {currency!r}"
            )
        # Patch 5: shared cash-mutation lock from PortfolioManager. None
        # means no concurrency protection (single-bot tests, legacy
        # callers); a real Lock is plumbed through bot_kwargs in normal
        # operation.
        self.cash_lock = cash_lock

        # Price tracking. last_price/reference_price are stored in the
        # bot's native currency (USD float for US, KRW int for KR);
        # cash and equity are always KRW int.
        self.reference_price: float | int | None = None
        self.last_price: float | int = 0
        self._price_history: list[float | int] = []
        self._sma_buf: collections.deque = collections.deque(
            maxlen=sma_period,
        )

        # Portfolio per bot
        self.cash: int = initial_cash
        self.position: int = 0
        self._highest_since_entry: int = 0

        # Daily-OHLCV ML gate state
        self._ohlcv_cache = get_cache()
        self._last_ml_check: float = 0.0
        self._cached_ml_signal: str = "HOLD"

        # Adaptive-threshold (Phase 1 EOD) state
        self.atr_period = 14
        self.pt_atr_mult = 2.0
        self.sl_atr_mult = 1.0
        self.chandelier_mult = 3.0
        self.breakout_atr_mult = 1.5
        self.entry_mode = entry_mode
        self.max_trades_per_day = max_trades_per_day
        self.min_holding_bars = 1
        self._trades_today = 0
        self._today_date: str = ""
        self._entry_bar_idx: int | None = None
        self._daily_ohlcv = None
        self._last_daily_refresh: float = 0.0
        self._entry_atr: float = 0.0

        self._running = False

    ML_CHECK_INTERVAL = 3600  # seconds between daily-OHLCV ML re-evaluations
    DAILY_REFRESH_INTERVAL = 3600  # seconds between daily OHLCV/indicator refresh

    # ── Properties ───────────────────────────────────────────────────

    @property
    def equity(self) -> int:
        """
        Current equity in KRW.

        Position value is mark-to-market in the bot's native currency
        (USD or KRW). For USD bots we convert to KRW using the cached
        FX rate, falling back to default_usdkrw if the cache is stale
        or zero.
        """
        if self.currency == "USD":
            fx = self._fx_to_krw()
            return int(self.cash + self.position * self.last_price * fx)
        return int(self.cash + self.position * self.last_price)

    def _fx_to_krw(self) -> float:
        """
        USD->KRW rate for currency conversion. Returns 1.0 for KRW bots.

        Reads ``client._cached_usdkrw``, falls back to
        ``client.default_usdkrw`` (and finally 1380.0) if the cache is
        zero, missing, or non-finite.
        """
        if self.currency != "USD":
            return 1.0
        fx = getattr(self.client, "_cached_usdkrw", 0) or 0
        if not fx or fx <= 0 or fx != fx:  # 0 / negative / NaN
            fx = getattr(self.client, "default_usdkrw", 0) or 1380.0
        return float(fx)

    def _fmt_price(self, p) -> str:
        """Format a price for logs (USD: 2dp, KRW: int)."""
        if self.currency == "USD":
            return f"{float(p):.2f}"
        return f"{int(p):,}"

    def _cash_lock_ctx(self):
        """
        Async context manager around cash mutations (Patch 5).

        Returns the shared PortfolioManager lock when one is plumbed
        through bot_kwargs; otherwise a no-op context manager so
        single-bot tests / legacy callers still work without the lock.
        """
        if self.cash_lock is not None:
            return self.cash_lock
        return _NullAsyncLock()

    def _now_str(self) -> str:
        return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

    # ── SMA ──────────────────────────────────────────────────────────

    def _get_sma(self) -> float | None:
        if len(self._sma_buf) < self.sma_period:
            return None
        return sum(self._sma_buf) / len(self._sma_buf)

    # ── ML Gate (daily OHLCV) ────────────────────────────────────────

    def _get_ml_signal(self) -> str:
        """
        Get ML signal based on daily OHLCV, cached for ML_CHECK_INTERVAL seconds.

        Returns 'BUY', 'SELL', or 'HOLD'. Falls back to 'HOLD' on any error
        so trades never depend on a silent inference failure.
        """
        now = time.time()
        if now - self._last_ml_check < self.ML_CHECK_INTERVAL:
            return self._cached_ml_signal

        ohlcv = self._ohlcv_cache.get(self.ticker)
        if ohlcv is None or len(ohlcv) < 60:
            self._cached_ml_signal = "HOLD"
        else:
            try:
                self._cached_ml_signal = self.predictor.predict_signal(ohlcv)
            except Exception as e:
                logger.warning("[%s] ML signal error: %s", self.ticker, e)
                self._cached_ml_signal = "HOLD"

        self._last_ml_check = now
        logger.info(
            "[%s] ML signal refreshed: %s",
            self.ticker, self._cached_ml_signal,
        )
        return self._cached_ml_signal

    # ── Daily-bar EOD state ──────────────────────────────────────────

    def _refresh_daily_data(self) -> None:
        """Reload daily OHLCV from cache and attach ATR/Donchian/breakout."""
        now = time.time()
        if now - self._last_daily_refresh < self.DAILY_REFRESH_INTERVAL:
            return

        from indicators import atr, donchian_channel, adaptive_breakout_level

        df = self._ohlcv_cache.get(self.ticker)
        if df is None or len(df) < 60:
            logger.warning(
                "[%s] insufficient daily data for indicators", self.ticker,
            )
            return

        df = df.copy()
        df["atr_14"] = atr(df, self.atr_period)
        df["donchian_high"], df["donchian_low"] = donchian_channel(df, 20)
        df["breakout_level"] = adaptive_breakout_level(
            df, self.breakout_atr_mult, self.atr_period,
        )
        self._daily_ohlcv = df
        self._last_daily_refresh = now

    def _is_eod_window(self) -> bool:
        """True if we are in the last 15 minutes before market close."""
        from config import MARKET_CLOSE
        now = datetime.now(KST)
        close_t = now.replace(
            hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1],
            second=0, microsecond=0,
        )
        minutes_to_close = (close_t - now).total_seconds() / 60
        return 0 < minutes_to_close <= 15

    def _reset_daily_counter(self) -> None:
        """Reset trade counter at the start of each trading day."""
        today = datetime.now(KST).strftime("%Y-%m-%d")
        if today != self._today_date:
            self._today_date = today
            self._trades_today = 0
            logger.info("[%s] New trading day: %s", self.ticker, today)

    def _evaluate_eod(self, price: int) -> str | None:
        """
        End-of-day evaluation using daily OHLCV + ATR-based rules.

        Entry (all must pass):
          1. price >= adaptive breakout level OR price > Donchian(20) high
          2. ML signal == BUY
          3. ATR/price < 5% (volatility cap)
          4. trade count < max_trades_per_day

        Exit (any triggers):
          1. Chandelier stop: price <= peak - 3*ATR
          2. Initial stop:    price <= entry - 1*ATR_at_entry
          3. ML signal == SELL
        """
        self._refresh_daily_data()
        self._reset_daily_counter()

        if self._daily_ohlcv is None or len(self._daily_ohlcv) < 30:
            return None

        latest = self._daily_ohlcv.iloc[-1]
        current_atr = float(latest["atr_14"])
        if not (current_atr > 0):
            return None

        # Exit path has priority over entry.
        if self.position > 0:
            self._highest_since_entry = max(self._highest_since_entry, price)
            chand_stop = (
                self._highest_since_entry - self.chandelier_mult * current_atr
            )
            if price <= chand_stop:
                logger.info(
                    "[%s] CHANDELIER EXIT: price=%s <= stop=%.2f "
                    "(peak=%s, ATR=%.2f)",
                    self.ticker, self._fmt_price(price), chand_stop,
                    self._fmt_price(self._highest_since_entry), current_atr,
                )
                return "SELL"

            if self._entry_atr > 0 and self.reference_price is not None:
                init_stop = (
                    self.reference_price - self.sl_atr_mult * self._entry_atr
                )
                if price <= init_stop:
                    logger.info(
                        "[%s] INITIAL STOP: price=%s <= entry_stop=%.2f",
                        self.ticker, self._fmt_price(price), init_stop,
                    )
                    return "SELL"

            ml_signal = self._get_ml_signal()
            if ml_signal == "SELL":
                logger.info("[%s] ML SELL signal", self.ticker)
                return "SELL"

            return None

        # Entry path
        if self._trades_today >= self.max_trades_per_day:
            return None

        if current_atr / price > 0.05:
            logger.debug(
                "[%s] Entry skipped: ATR/price=%.3f too high",
                self.ticker, current_atr / price,
            )
            return None

        breakout_level = float(latest["breakout_level"])
        donchian_high = float(latest["donchian_high"])
        breakout = (price >= breakout_level) or (price > donchian_high)
        if not breakout:
            return None

        ml_signal = self._get_ml_signal()
        if ml_signal != "BUY":
            logger.debug(
                "[%s] Entry blocked by ML (signal=%s)",
                self.ticker, ml_signal,
            )
            return None

        logger.info(
            "[%s] EOD BUY SIGNAL: price=%s, breakout=%.2f, donchian=%.2f, "
            "ATR=%.2f, ML=BUY",
            self.ticker, self._fmt_price(price),
            breakout_level, donchian_high, current_atr,
        )
        self._entry_atr = current_atr
        return "BUY"

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

        # ── Trailing stop for open long positions ────────────────
        # Risk management runs independently of the ML gate.
        if self.position > 0:
            self._highest_since_entry = max(
                self._highest_since_entry, price,
            )
            trail_price = int(
                self._highest_since_entry * (1 - self.trailing_stop_pct)
            )
            if price <= trail_price:
                logger.info(
                    "[%s] TRAILING STOP: %s ≤ %s (peak=%s, trail=%.1f%%)",
                    self.ticker,
                    self._fmt_price(price),
                    self._fmt_price(trail_price),
                    self._fmt_price(self._highest_since_entry),
                    self.trailing_stop_pct * 100,
                )
                return "SELL"

        # ── BUY signal ───────────────────────────────────────────
        if pct >= self.threshold:
            if sma is not None and price <= sma:
                logger.debug(
                    "[%s] BUY suppressed: price %s ≤ SMA %.2f",
                    self.ticker, self._fmt_price(price), sma,
                )
                return None
            ml_signal = self._get_ml_signal()
            if ml_signal != "BUY":
                logger.debug(
                    "[%s] BUY blocked by ML gate (signal=%s)",
                    self.ticker, ml_signal,
                )
                return None
            sma_str = f"{sma:,.0f}" if sma else "warmup"
            logger.info(
                "[%s] BUY SIGNAL: +%.2f%%, SMA=%s, ML=BUY",
                self.ticker, pct * 100, sma_str,
            )
            return "BUY"

        # ── SELL signal ──────────────────────────────────────────
        if pct <= -self.threshold:
            if sma is not None and price >= sma:
                logger.debug(
                    "[%s] SELL suppressed: price %s ≥ SMA %.2f",
                    self.ticker, self._fmt_price(price), sma,
                )
                return None
            logger.info(
                "[%s] SELL SIGNAL: %.2f%%",
                self.ticker, pct * 100,
            )
            return "SELL"

        return None

    # ── Order Execution ──────────────────────────────────────────────

    async def _execute(self, side: str, price) -> None:
        # ── Decide qty BEFORE placing the order (Patch 2) ────────
        # BUY:  size to ~95% of cash (5% buffer for cost/slippage drift),
        #       refuse to add to an existing position (no averaging up).
        # SELL: full liquidation — sell whatever position exists.
        fx = self._fx_to_krw()  # 1.0 for KRW bots

        if side == "BUY":
            if self.position > 0:
                logger.info(
                    "[%s] BUY skipped: already holding %d shares "
                    "(no averaging up)",
                    self.ticker, self.position,
                )
                return

            usable_cash_krw = int(self.cash * 0.95)
            # Slippage buffer (0.1%) so the request doesn't get clipped
            # at fill time to one share less than we expected.
            cost_per_share_native = price * (1 + self.commission + 0.001)
            cost_per_share_krw = cost_per_share_native * fx
            if cost_per_share_krw <= 0:
                logger.warning(
                    "[%s] BUY skipped: invalid cost-per-share %.2f",
                    self.ticker, cost_per_share_krw,
                )
                return

            qty_to_trade = int(usable_cash_krw // cost_per_share_krw)
            if qty_to_trade < 1:
                logger.info(
                    "[%s] BUY skipped: insufficient cash "
                    "(have=%d KRW, need ~%d KRW for 1 share)",
                    self.ticker, self.cash, int(cost_per_share_krw),
                )
                return
        else:  # SELL
            if self.position < 1:
                logger.info(
                    "[%s] SELL skipped: no open position", self.ticker,
                )
                return
            qty_to_trade = self.position

        ml_score = self.predictor.score(self._price_history)
        try:
            result = await self.client.place_order(
                ticker=self.ticker, qty=qty_to_trade, side=side,
                order_type="market",
            )
            output = result.get("output", {})
            order_no = output.get("ODNO", "N/A")

            # KIS order response does not include the actual fill price.
            # Poll the fill-query endpoint so slippage is reflected in logs.
            fill_price = price
            if order_no != "N/A" and hasattr(self.client, "get_order_fill"):
                await asyncio.sleep(0.5)
                try:
                    ccld = await self.client.get_order_fill(order_no)
                    if ccld.get("avg_fill_price", 0) > 0:
                        fill_price = ccld["avg_fill_price"]
                except Exception as e:
                    logger.warning(
                        "[%s] Fill price query failed: %s "
                        "(using request price)",
                        self.ticker, e,
                    )

            sell_tax = self.tax_kr if self.currency == "KRW" else self.tax_us

            # Patch 5: serialize cash mutations across all bots. Lock is
            # held only across pure sync arithmetic — no awaits inside —
            # so concurrent bots aren't serialized on each other's
            # network calls.
            async with self._cash_lock_ctx():
                if side == "BUY":
                    cost_native = (
                        fill_price * qty_to_trade * (1 + self.commission)
                    )
                    cost_krw = int(cost_native * fx)
                    self.cash -= cost_krw
                    self.position += qty_to_trade
                    self._highest_since_entry = fill_price
                    if not self._entry_atr:
                        self._entry_atr = fill_price * 0.02
                else:
                    proceeds_native = (
                        fill_price * qty_to_trade
                        * (1 - self.commission - sell_tax)
                    )
                    proceeds_krw = int(proceeds_native * fx)
                    self.cash += proceeds_krw
                    # Always full liquidation on SELL.
                    self.position = 0
                    self._highest_since_entry = 0
                    self._entry_atr = 0.0

                self.reference_price = fill_price
                self._trades_today += 1

            self._append_trade_csv(
                side, price, fill_price, order_no, ml_score, qty_to_trade,
            )
            trade_logger.info(
                "%s %s x%d @ %s (fill=%s %s, fx=%.1f) "
                "cash=%d KRW pos=%d equity=%d KRW ML=%.3f",
                side, self.ticker, qty_to_trade,
                self._fmt_price(price), self._fmt_price(fill_price),
                self.currency, fx,
                self.cash, self.position, self.equity, ml_score,
            )

        except Exception as e:
            logger.error("[%s] Order failed: %s", self.ticker, e)

    # ── CSV Logging ──────────────────────────────────────────────────

    def _append_trade_csv(
        self, side: str, price, fill_price,
        order_no: str, ml_score: float, qty: int | None = None,
    ) -> None:
        row = [
            self._now_str(), self.ticker, side, price, fill_price,
            qty if qty is not None else self.qty,
            order_no, self.equity, f"{ml_score:.3f}",
        ]
        with open(TRADE_LOG, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _append_equity_csv(self, price) -> None:
        row = [self._now_str(), self.ticker, price, self.equity]
        with open(EQUITY_LOG, "a", newline="") as f:
            csv.writer(f).writerow(row)

    # ── Main Loop ────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Async main loop.

        In EOD mode (default): poll slowly and only evaluate/trade during
        the last 15 minutes before market close. In intraday mode (legacy):
        10-second polling with momentum threshold — backward compat only.
        Runs until .stop() is called or the task is cancelled.
        """
        self._running = True
        logger.info(
            "[%s] Bot started (mode=%s, max_trades/day=%d, cash=%s)",
            self.ticker, self.entry_mode, self.max_trades_per_day,
            f"{self.cash:,}",
        )

        # Initial price fetch
        try:
            price = await self.client.get_current_price(self.ticker)
            self.reference_price = price
            self.last_price = price
            self._price_history.append(price)
            self._sma_buf.append(price)
            logger.info(
                "[%s] Initial price: %s %s",
                self.ticker, self._fmt_price(price), self.currency,
            )
        except Exception as e:
            logger.error("[%s] Failed to start: %s", self.ticker, e)
            return

        # EOD mode polls slowly — the signal only runs in the 15-min window.
        effective_interval = (
            60 if self.entry_mode == "eod" else self.poll_interval
        )

        while self._running:
            try:
                price = await self.client.get_current_price(self.ticker)
                self.last_price = price
                self._price_history.append(price)
                self._sma_buf.append(price)

                if self.entry_mode == "eod":
                    if self._is_eod_window():
                        signal = self._evaluate_eod(price)
                        if signal:
                            await self._execute(signal, price)
                else:
                    signal = self._evaluate(price)
                    if signal:
                        await self._execute(signal, price)

                self._append_equity_csv(price)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[%s] Poll error: %s", self.ticker, e)

            try:
                await asyncio.sleep(effective_interval)
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
