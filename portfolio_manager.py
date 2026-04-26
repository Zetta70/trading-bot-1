"""
Portfolio Manager — orchestrates the full trading pipeline.

Responsibilities:
  1. Manage TradingBot lifecycle (start / stop / restart).
  2. Run MarketScanner periodically to discover new tickers.
  3. Gate new tickers through MLPredictor before activation.
  4. Track global equity and enforce the kill-switch drawdown limit.
  5. Save / load portfolio state to state.json for crash recovery.
  6. Pause during non-market hours (KST 15:40–08:50) in REAL mode.
"""

import asyncio
import csv
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from config import KST, MARKET_OPEN, MARKET_CLOSE
from market_scanner import MarketScanner
from ml_predictor import MLPredictor
from risk_manager import RiskManager
from trading_bot import TradingBot, EQUITY_LOG, EQUITY_HEADER

logger = logging.getLogger(__name__)

STATE_FILE = "state.json"


class KillSwitchError(Exception):
    """Raised when the global drawdown limit is breached."""


class PortfolioManager:
    """
    Top-level orchestrator for multi-ticker paper trading.

    Holds a pool of TradingBot instances, each running as an
    independent asyncio task.
    """

    def __init__(
        self,
        client,
        scanner: MarketScanner,
        predictor: MLPredictor,
        *,
        initial_cash: int = 10_000_000,
        max_drawdown_pct: float = 0.03,
        max_active_bots: int = 10,
        scan_interval: int = 900,
        skip_market_hours: bool = False,
        bot_kwargs: dict | None = None,
    ):
        self.client = client
        self.scanner = scanner
        self.predictor = predictor

        self.initial_cash = initial_cash
        self.env_initial_cash = initial_cash  # Patch 3: cap for sync
        self.max_drawdown_pct = max_drawdown_pct
        self.max_active_bots = max_active_bots
        self.scan_interval = scan_interval
        self.skip_market_hours = skip_market_hours
        self._bot_kwargs = bot_kwargs or {}

        # Global state
        self._remaining_cash: int = initial_cash
        self._peak_equity: int = initial_cash
        self._running = False
        self._last_sync_time: str | None = None  # ISO-8601, set by sync_balance
        self._state_loaded = False  # set by load_state(), read by sync_balance

        # Bot registry
        self.bots: dict[str, TradingBot] = {}
        self._tasks: dict[str, asyncio.Task] = {}

        # Phase 3: portfolio-level risk controller
        self.risk_manager = RiskManager(
            max_sector_exposure=float(os.getenv("MAX_SECTOR_EXPOSURE", "0.30")),
            portfolio_stop_loss=float(os.getenv("PORTFOLIO_STOP", "-0.05")),
            portfolio_kill=float(os.getenv("PORTFOLIO_KILL", "-0.10")),
        )
        self._load_sector_mapping()
        self._last_equity_for_return: int = initial_cash

    def _load_sector_mapping(self) -> None:
        path = Path("data/sector_map_kr.csv")
        if not path.exists():
            logger.info("No sector_map_kr.csv found; sector limits disabled")
            return
        import csv as _csv
        with open(path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                self.risk_manager.register_sector(row["ticker"], row["sector"])
        logger.info(
            "Loaded sector mapping for %d tickers",
            len(self.risk_manager._sector_map),
        )

    # ── Equity ───────────────────────────────────────────────────────

    @property
    def total_equity(self) -> int:
        return self._remaining_cash + sum(
            b.equity for b in self.bots.values()
        )

    # ── Bot Lifecycle ────────────────────────────────────────────────

    def _cash_per_bot(self) -> int:
        return self.initial_cash // self.max_active_bots

    async def add_bot(self, ticker: str) -> bool:
        """Create and start a bot for ticker. Returns False if skipped."""
        if ticker in self.bots:
            return False
        if len(self.bots) >= self.max_active_bots:
            logger.warning("Max bots (%d) reached. Skipping %s.",
                           self.max_active_bots, ticker)
            return False

        # Phase 3: Sector exposure check
        current_positions = {
            t: bot.position * bot.last_price for t, bot in self.bots.items()
        }
        if not self.risk_manager.check_sector_concurrence(
            ticker, current_positions, self.total_equity,
        ):
            logger.warning(
                "[%s] Rejected: sector exposure would exceed %.0f%%",
                ticker, self.risk_manager.max_sector_exposure * 100,
            )
            return False

        # Phase 3: VaR breach check
        if self.risk_manager.is_var_breached():
            logger.warning(
                "[%s] Rejected: portfolio VaR limit breached", ticker,
            )
            return False

        # Phase 3: Deployment scale check (drawdown-based)
        if self._peak_equity > 0:
            dd = (self.total_equity - self._peak_equity) / self._peak_equity
        else:
            dd = 0.0
        scale = self.risk_manager.compute_deployment_scale(
            current_vol_index=None,
            current_drawdown=dd,
        )
        if scale <= 0:
            logger.critical(
                "Deployment scale = 0 — rejecting all new positions"
            )
            return False

        alloc = min(self._cash_per_bot(), self._remaining_cash)
        if alloc <= 0:
            logger.warning("No cash left to allocate for %s.", ticker)
            return False

        self._remaining_cash -= alloc

        bot = TradingBot(
            client=self.client,
            ticker=ticker,
            predictor=self.predictor,
            initial_cash=alloc,
            **self._bot_kwargs,
        )
        self.bots[ticker] = bot
        self._tasks[ticker] = asyncio.create_task(
            bot.run(), name=f"bot-{ticker}",
        )
        logger.info(
            "Started bot for %s (allocated %s KRW)", ticker, f"{alloc:,}",
        )
        return True

    def _stop_all_bots(self) -> None:
        for bot in self.bots.values():
            bot.stop()
        for task in self._tasks.values():
            task.cancel()

    # ── Market Hours ─────────────────────────────────────────────────

    def _is_market_open(self) -> bool:
        if self.skip_market_hours:
            return True
        now = datetime.now(KST)
        if now.weekday() >= 5:  # Weekend
            return False
        open_time = now.replace(
            hour=MARKET_OPEN[0], minute=MARKET_OPEN[1],
            second=0, microsecond=0,
        )
        close_time = now.replace(
            hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1],
            second=0, microsecond=0,
        )
        return open_time <= now <= close_time

    def _next_market_open(self) -> datetime:
        now = datetime.now(KST)
        candidate = now.replace(
            hour=MARKET_OPEN[0], minute=MARKET_OPEN[1],
            second=0, microsecond=0,
        )
        if candidate <= now:
            candidate += timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate

    async def _wait_for_market(self) -> None:
        """Sleep until next market open, checking every 5 minutes."""
        while not self._is_market_open() and self._running:
            next_open = self._next_market_open()
            wait_sec = (next_open - datetime.now(KST)).total_seconds()
            logger.info(
                "Market closed. Next open: %s KST (%.0f min)",
                next_open.strftime("%Y-%m-%d %H:%M"), wait_sec / 60,
            )
            self.save_state()
            await asyncio.sleep(min(wait_sec, 300))

    # ── Kill Switch ──────────────────────────────────────────────────

    def _check_drawdown(self) -> None:
        equity = self.total_equity
        self._peak_equity = max(self._peak_equity, equity)

        if self._peak_equity <= 0:
            return

        drawdown = (self._peak_equity - equity) / self._peak_equity
        if drawdown >= self.max_drawdown_pct:
            logger.critical(
                "KILL SWITCH: drawdown %.2f%% ≥ %.2f%%. "
                "Peak=%s, Current=%s",
                drawdown * 100, self.max_drawdown_pct * 100,
                f"{self._peak_equity:,}", f"{equity:,}",
            )
            raise KillSwitchError(
                f"Drawdown {drawdown:.2%} exceeds {self.max_drawdown_pct:.2%}"
            )

    # ── Balance Sync (Patch 3) ───────────────────────────────────────

    SYNC_TTL_HOURS = 24

    async def sync_balance(self) -> None:
        """
        Reconcile bot working capital with the broker's actual deposit.

        Contract:
          * Calls ``client.get_balance()`` and reads ``deposit`` (KRW).
          * Sets ``self.initial_cash = min(actual_krw, env_initial_cash)``.
          * If no state was loaded, sets ``_remaining_cash`` to the new
            initial_cash. If state WAS loaded, leaves ``_remaining_cash``
            alone (its value reflects already-allocated capital).
          * Stores ``last_sync_time`` (ISO-8601 KST). Skips the API call
            if a previous sync is younger than ``SYNC_TTL_HOURS``.
          * On any failure (network, missing field, etc.) falls back to
            the .env value with a WARNING; never blocks startup.

        Must be called after ``load_state()`` and before any ``add_bot()``.
        """
        # 24-hour TTL: skip the network call when state has a recent sync.
        if self._last_sync_time:
            try:
                last = datetime.fromisoformat(self._last_sync_time)
                age_hours = (
                    datetime.now(KST) - last
                ).total_seconds() / 3600
                if age_hours < self.SYNC_TTL_HOURS:
                    logger.info(
                        "Account sync: skipped — last sync %.1fh ago "
                        "(< %dh TTL)",
                        age_hours, self.SYNC_TTL_HOURS,
                    )
                    return
            except (ValueError, TypeError):
                pass  # corrupted timestamp — fall through and re-sync

        env_limit = self.env_initial_cash
        try:
            balance = await self.client.get_balance()
            actual_krw = int(balance.get("deposit", 0))
        except Exception as e:
            logger.warning(
                "Account sync failed: %s. Falling back to .env "
                "INITIAL_CASH=%s KRW.",
                e, f"{env_limit:,}",
            )
            return  # leave initial_cash / _remaining_cash unchanged

        if actual_krw <= 0:
            logger.warning(
                "Account sync: broker reported deposit=%s KRW (zero or "
                "missing). Falling back to .env INITIAL_CASH=%s KRW.",
                f"{actual_krw:,}", f"{env_limit:,}",
            )
            return

        new_initial = min(actual_krw, env_limit)
        logger.info(
            "Account sync: KIS balance=%s KRW, .env limit=%s KRW "
            "→ using %s KRW",
            f"{actual_krw:,}", f"{env_limit:,}", f"{new_initial:,}",
        )
        self.initial_cash = new_initial
        self._peak_equity = max(self._peak_equity, new_initial)
        if not self._state_loaded:
            self._remaining_cash = new_initial
            self._last_equity_for_return = new_initial
        self._last_sync_time = datetime.now(KST).isoformat()

    # ── State Persistence ────────────────────────────────────────────

    def save_state(self) -> None:
        state = {
            "timestamp": datetime.now(KST).isoformat(),
            "remaining_cash": self._remaining_cash,
            "peak_equity": self._peak_equity,
            "total_equity": self.total_equity,
            "last_sync_time": self._last_sync_time,
            "bots": {
                ticker: bot.get_state()
                for ticker, bot in self.bots.items()
            },
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.debug("State saved to %s", STATE_FILE)

    def load_state(self) -> bool:
        """Load saved state. Returns True if state was restored."""
        if not Path(STATE_FILE).exists():
            return False

        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load state: %s", e)
            return False

        self._remaining_cash = state.get(
            "remaining_cash", self.initial_cash,
        )
        self._peak_equity = state.get("peak_equity", self.initial_cash)
        self._last_sync_time = state.get("last_sync_time")

        restored = 0
        for ticker, bot_state in state.get("bots", {}).items():
            bot = TradingBot(
                client=self.client,
                ticker=ticker,
                predictor=self.predictor,
                initial_cash=0,
                **self._bot_kwargs,
            )
            bot.load_state(bot_state)
            self.bots[ticker] = bot
            restored += 1

        logger.info(
            "State restored: %d bots, equity=%s KRW (last sync: %s)",
            restored, f"{self.total_equity:,}",
            self._last_sync_time or "never",
        )
        self._state_loaded = True
        return True

    # ── Concurrent Tasks ─────────────────────────────────────────────

    async def _scanner_loop(self) -> None:
        """Periodically scan for new tickers and add qualifying ones."""
        while self._running:
            await asyncio.sleep(self.scan_interval)
            if not self._is_market_open():
                continue

            try:
                candidates = await self.scanner.scan()
                from ohlcv_cache import get_cache
                cache = get_cache()

                for cand in candidates:
                    ticker = cand["ticker"]
                    if ticker in self.bots:
                        continue

                    # ML pre-screen using DAILY OHLCV (not live ticks).
                    # Tick-frequency features make RSI/volatility meaningless.
                    ohlcv = cache.get(ticker)
                    if ohlcv is None or len(ohlcv) < 60:
                        logger.info(
                            "Scanner: %s skipped — insufficient daily OHLCV for ML",
                            ticker,
                        )
                        continue

                    if self.predictor._ml_model is not None:
                        signal = self.predictor.predict_signal(ohlcv)
                        if signal == "BUY":
                            logger.info(
                                "Scanner: %s passed ML gate (BUY). Adding bot.",
                                ticker,
                            )
                            await self.add_bot(ticker)
                        else:
                            logger.info(
                                "Scanner: %s rejected by ML (signal=%s)",
                                ticker, signal,
                            )
                    else:
                        # Rule-based fallback on daily closes
                        closes = ohlcv["close"].astype(int).tolist()
                        score = self.predictor.score(closes)
                        threshold = self._bot_kwargs.get(
                            "bullish_threshold", 0.75,
                        )
                        if score >= threshold:
                            logger.info(
                                "Scanner: %s passed rule-based gate "
                                "(%.3f >= %.3f). Adding bot.",
                                ticker, score, threshold,
                            )
                            await self.add_bot(ticker)
                        else:
                            logger.info(
                                "Scanner: %s rejected by rule-based "
                                "(%.3f < %.3f)",
                                ticker, score, threshold,
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scanner error: %s", e)

    async def _equity_monitor(self) -> None:
        """Check total equity and enforce kill switch every cycle."""
        while self._running:
            await asyncio.sleep(self._bot_kwargs.get("poll_interval", 10))
            try:
                self._check_drawdown()

                # Phase 3: feed daily-return buffer for VaR tracking.
                # (Every cycle here — the RiskManager only triggers once
                # the buffer holds enough samples; refine to true daily
                # deltas if a coarser cadence is preferred.)
                cur_equity = self.total_equity
                if self._last_equity_for_return > 0:
                    daily_ret = (
                        cur_equity - self._last_equity_for_return
                    ) / self._last_equity_for_return
                    self.risk_manager.update_daily_return(daily_ret)
                self._last_equity_for_return = cur_equity

                # Log portfolio-level equity
                now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
                with open(EQUITY_LOG, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [now, "PORTFOLIO", "", cur_equity]
                    )
            except KillSwitchError:
                raise
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Equity monitor error: %s", e)

    async def _state_saver(self) -> None:
        """Save state to disk every 60 seconds."""
        while self._running:
            await asyncio.sleep(60)
            try:
                self.save_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("State save error: %s", e)

    async def _bot_supervisor(self) -> None:
        """Monitor bot tasks and log if any crash unexpectedly."""
        while self._running:
            await asyncio.sleep(30)
            for ticker, task in list(self._tasks.items()):
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc:
                        logger.error(
                            "Bot %s crashed: %s. Restarting...",
                            ticker, exc,
                        )
                        bot = self.bots[ticker]
                        bot._running = False
                        self._tasks[ticker] = asyncio.create_task(
                            bot.run(), name=f"bot-{ticker}",
                        )

    # ── Main Entry ───────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Start all subsystems and run until killed or kill-switched.
        """
        self._running = True
        logger.info(
            "Portfolio Manager started. Equity=%s KRW, Bots=%d",
            f"{self.total_equity:,}", len(self.bots),
        )

        # Start bot tasks for any pre-loaded bots (from state)
        for ticker, bot in self.bots.items():
            if ticker not in self._tasks:
                self._tasks[ticker] = asyncio.create_task(
                    bot.run(), name=f"bot-{ticker}",
                )

        try:
            while self._running:
                # Gate on market hours
                if not self._is_market_open():
                    self._stop_all_bots()
                    self._tasks.clear()
                    await self._wait_for_market()
                    if not self._running:
                        break
                    # Restart bots after market opens
                    for ticker, bot in self.bots.items():
                        bot._running = False
                        self._tasks[ticker] = asyncio.create_task(
                            bot.run(), name=f"bot-{ticker}",
                        )
                    logger.info("Market OPEN. %d bots restarted.", len(self.bots))

                # Run all subsystems concurrently
                await asyncio.gather(
                    self._scanner_loop(),
                    self._equity_monitor(),
                    self._state_saver(),
                    self._bot_supervisor(),
                )

        except KillSwitchError:
            logger.critical("Portfolio Manager: KILL SWITCH activated.")
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            self._stop_all_bots()
            self.save_state()
            logger.info(
                "Portfolio Manager stopped. Final equity: %s KRW",
                f"{self.total_equity:,}",
            )

    def stop(self) -> None:
        self._running = False
