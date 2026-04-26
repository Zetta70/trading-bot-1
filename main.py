"""
Entry point for the KIS Trading Bot — Dual Market (KR + US).

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │                    main.py                           │
  │  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │
  │  │ KR Session│  │ US Session│  │  Idle Tasks        │ │
  │  │ 08:50~    │  │ 23:30~   │  │  (model retrain,   │ │
  │  │ 15:40 KST │  │ 06:00 KST│  │   data backup)     │ │
  │  └──────────┘  └──────────┘  └────────────────────┘ │
  └──────────────────────────────────────────────────────┘

Scheduler (all times in KST):
  09:00–15:30 KST  → Korean market session (KISClient)
  15:40–23:30 KST  → Idle window: model retrain + data backup
  23:30–06:00 KST  → US market session (KISUSClient)
  06:00–08:50 KST  → Idle window: sleep

DST handling: US market times are derived from America/New_York
  timezone (zoneinfo), which auto-handles EST/EDT. The KST
  equivalents shift accordingly (~23:30 or ~22:30 KST).
"""

import asyncio
import random
import logging
import signal
from datetime import datetime, timedelta
from pathlib import Path

from config import Config, KST, EST, MARKET_OPEN, MARKET_CLOSE, US_MARKET_OPEN, US_MARKET_CLOSE
from logger_setup import setup_logging
from kis_client import KISClient
from market_scanner import MarketScanner, CrossSectionalScanner
from ml_predictor import MLPredictor
from portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Market Time Helpers
# ═══════════════════════════════════════════════════════════════════════

def is_kr_market_open() -> bool:
    """Check if Korean market is currently open (KST)."""
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(
        hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0,
    )
    close_t = now.replace(
        hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0,
    )
    return open_t <= now <= close_t


def is_us_market_open() -> bool:
    """Check if US market is currently open (ET, DST-aware)."""
    now_et = datetime.now(EST)
    if now_et.weekday() >= 5:
        return False
    open_t = now_et.replace(
        hour=US_MARKET_OPEN[0], minute=US_MARKET_OPEN[1],
        second=0, microsecond=0,
    )
    close_t = now_et.replace(
        hour=US_MARKET_CLOSE[0], minute=US_MARKET_CLOSE[1],
        second=0, microsecond=0,
    )
    return open_t <= now_et <= close_t


def next_kr_market_open() -> datetime:
    """Return the next KR market open time as KST datetime."""
    now = datetime.now(KST)
    candidate = now.replace(
        hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0,
    )
    if candidate <= now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def next_us_market_open() -> datetime:
    """Return the next US market open time as KST datetime."""
    now_et = datetime.now(EST)
    candidate = now_et.replace(
        hour=US_MARKET_OPEN[0], minute=US_MARKET_OPEN[1],
        second=0, microsecond=0,
    )
    if candidate <= now_et:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    # Convert to KST for unified scheduling
    return candidate.astimezone(KST)


def _seconds_until(target_kst: datetime) -> float:
    """Seconds from now until a KST target time."""
    return max(0, (target_kst - datetime.now(KST)).total_seconds())


# ═══════════════════════════════════════════════════════════════════════
# Session Runners
# ═══════════════════════════════════════════════════════════════════════

async def run_kr_session(config: Config, stop_event: asyncio.Event) -> None:
    """Run the Korean market trading session."""
    if not config.tickers_kr:
        logger.info("No KR tickers configured. Skipping KR session.")
        return

    logger.info("═" * 50)
    logger.info("  KR SESSION START")
    logger.info("═" * 50)

    client = KISClient(
        api_key=config.api_key,
        api_secret=config.api_secret,
        acc_no=config.acc_no,
        run_mode=config.run_mode,
    )

    predictor = MLPredictor.from_config(config)

    if config.use_cross_sectional_scanner:
        def _load_kr_universe():
            import csv as _csv
            path = Path("data/universe_kr.csv")
            if path.exists():
                with open(path, newline="", encoding="utf-8") as f:
                    return [row["ticker"] for row in _csv.DictReader(f)]
            return config.tickers_kr

        scanner = CrossSectionalScanner(
            predictor=predictor,
            universe_loader=_load_kr_universe,
            top_n=config.cs_scanner_top_n,
        )
    else:
        scanner = MarketScanner(
            client, mock_mode=(config.run_mode == "MOCK"),
        )

    bot_kwargs = {
        # Patch 2: qty is no longer a kwarg — _execute sizes dynamically.
        "threshold": config.threshold,
        "poll_interval": config.poll_interval,
        "sma_period": config.sma_period,
        "bullish_threshold": config.bullish_threshold,
        "trailing_stop_pct": config.trailing_stop_pct,
        "commission": 0.00015,
        "tax_kr": 0.0023,
        "tax_us": 0.0,
        "entry_mode": config.entry_mode,
        "max_trades_per_day": config.max_trades_per_day,
        "currency": "KRW",
    }

    portfolio = PortfolioManager(
        client=client,
        scanner=scanner,
        predictor=predictor,
        initial_cash=config.initial_cash,
        max_drawdown_pct=config.max_drawdown_pct,
        max_active_bots=config.max_active_bots,
        scan_interval=config.scan_interval,
        skip_market_hours=False,
        bot_kwargs=bot_kwargs,
    )

    if portfolio.load_state():
        logger.info("KR: Resumed from saved state.")

    # Patch 3: reconcile working capital with the broker before sizing bots.
    await portfolio.sync_balance()

    for ticker in config.tickers_kr:
        await portfolio.add_bot(ticker)

    try:
        await portfolio.run()
    finally:
        await client.close()
        logger.info("KR SESSION END")


async def run_us_session(config: Config, stop_event: asyncio.Event) -> None:
    """
    Run the US market trading session.

    Uses KISUSClient for order execution. The trading logic reuses
    the same PortfolioManager/TradingBot architecture but with the
    US client adapter.
    """
    if not config.tickers_us:
        logger.info("No US tickers configured. Skipping US session.")
        return

    logger.info("═" * 50)
    logger.info("  US SESSION START")
    logger.info("═" * 50)

    from api.kis_us_api import KISUSClient

    us_client = KISUSClient(
        api_key=config.us_api_key,
        api_secret=config.us_api_secret,
        acc_no=config.us_acc_no,
        run_mode=config.run_mode,
        default_usdkrw=config.default_usdkrw,
    )

    # Adapter: wrap KISUSClient to match the KISClient interface
    # expected by TradingBot (get_current_price, place_order)
    adapter = _USClientAdapter(us_client)

    scanner = MarketScanner(adapter, mock_mode=True)  # US scanner = mock for now
    predictor = MLPredictor.from_config(config)

    us_cash = int(config.initial_cash * config.us_allocation_pct)
    bot_kwargs = {
        # Patch 2: qty is no longer a kwarg — _execute sizes dynamically.
        "threshold": config.threshold,
        "poll_interval": config.poll_interval,
        "sma_period": config.sma_period,
        "bullish_threshold": config.bullish_threshold,
        "trailing_stop_pct": config.trailing_stop_pct,
        "commission": 0.00015,
        "tax_kr": 0.0023,
        "tax_us": 0.0,
        "entry_mode": config.entry_mode,
        "max_trades_per_day": config.max_trades_per_day,
        "currency": "USD",
    }

    portfolio = PortfolioManager(
        client=adapter,
        scanner=scanner,
        predictor=predictor,
        initial_cash=us_cash,
        max_drawdown_pct=config.max_drawdown_pct,
        max_active_bots=min(len(config.tickers_us), config.max_active_bots),
        scan_interval=config.scan_interval,
        skip_market_hours=True,  # We handle market hours in the scheduler
        bot_kwargs=bot_kwargs,
    )

    # Patch 3: reconcile working capital (USD deposit -> KRW via FX).
    await portfolio.sync_balance()

    for ticker in config.tickers_us:
        await portfolio.add_bot(ticker)

    # Run until US market closes
    try:
        run_task = asyncio.create_task(portfolio.run())

        while not stop_event.is_set() and is_us_market_open():
            await asyncio.sleep(30)

        portfolio.stop()
        await asyncio.sleep(2)  # Grace period
        if not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

    finally:
        portfolio.save_state()
        await us_client.close()
        logger.info("US SESSION END")


class _USClientAdapter:
    """
    Adapter exposing the KISUSClient via the KISClient interface.

    Currency contract (Patch 1)
    ---------------------------
    The bot is currency-aware: ``get_current_price`` and
    ``get_order_fill["avg_fill_price"]`` return raw USD floats. The
    adapter no longer multiplies prices by 100 — the bot's
    ``self._fx_to_krw()`` now handles all currency conversion. The
    adapter forwards ``_cached_usdkrw`` and ``default_usdkrw`` so the
    bot can read FX rate without knowing the underlying client class.
    """

    def __init__(self, us_client):
        self._us = us_client

    @property
    def _cached_usdkrw(self) -> float:
        return self._us._cached_usdkrw

    @property
    def default_usdkrw(self) -> float:
        return self._us.default_usdkrw

    async def get_current_price(self, ticker: str) -> float:
        """Return the latest USD price as a float."""
        return float(await self._us.get_us_price(ticker))

    async def place_order(
        self,
        ticker: str,
        qty: int,
        side: str,
        order_type: str = "market",
        price: float = 0,
    ) -> dict:
        """Place US order. ``price`` is USD (0 for market orders)."""
        return await self._us.place_us_order(
            ticker, qty, side, order_type, float(price),
        )

    async def get_order_fill(self, order_no: str) -> dict:
        """Proxy to KISUSClient. Returns USD float ``avg_fill_price``."""
        return await self._us.get_order_fill(order_no)

    async def get_balance(self) -> dict:
        """
        KISClient-shaped balance for the PortfolioManager.

        The underlying call is ``KISUSClient.get_us_balance`` which
        returns USD fields. This adapter converts the deposit and
        eval totals to KRW using the cached FX rate so PortfolioManager
        can compare against ``INITIAL_CASH`` (KRW) without knowing the
        underlying market.
        """
        bal = await self._us.get_us_balance()
        fx = float(self._us._cached_usdkrw or self._us.default_usdkrw or 1380.0)
        deposit_usd = float(bal.get("deposit_usd", 0.0))
        total_eval_usd = float(bal.get("total_eval_usd", 0.0))
        return {
            "deposit": int(deposit_usd * fx),
            "deposit_usd": deposit_usd,
            "total_eval": int(total_eval_usd * fx),
            "fx_rate": fx,
            "holdings": bal.get("holdings", []),
        }

    async def close(self) -> None:
        await self._us.close()


# ═══════════════════════════════════════════════════════════════════════
# Idle Tasks (between sessions)
# ═══════════════════════════════════════════════════════════════════════

async def run_idle_tasks(config: Config) -> None:
    """
    Execute maintenance tasks during the idle window between KR close and US open.

    Tasks:
      1. Model retraining (if backtest module is available)
      2. State/log backup
    """
    logger.info("── Idle Window: Running maintenance tasks ──")

    # Task 1: Log rotation / backup summary
    try:
        from pathlib import Path
        import shutil
        from datetime import datetime as dt

        backup_dir = Path("backups") / dt.now(KST).strftime("%Y%m%d")
        backup_dir.mkdir(parents=True, exist_ok=True)

        for log_file in Path("logs").glob("*.log"):
            dest = backup_dir / log_file.name
            if not dest.exists():
                shutil.copy2(log_file, dest)

        state_file = Path("state.json")
        if state_file.exists():
            shutil.copy2(state_file, backup_dir / "state.json")

        logger.info("Backup complete → %s/", backup_dir)
    except Exception as e:
        logger.warning("Backup failed: %s", e)

    # Task 2: Model retrain hint
    logger.info(
        "Tip: Run 'python -m backtest.run_backtest --save-model models/lgbm_v1.pkl' "
        "to retrain the ML model with latest data."
    )


# ═══════════════════════════════════════════════════════════════════════
# Main Scheduler
# ═══════════════════════════════════════════════════════════════════════

async def scheduler(config: Config, stop_event: asyncio.Event) -> None:
    """
    Dual-market scheduler loop.

    Continuously checks which market is open and runs the appropriate session.
    During idle windows, runs maintenance tasks.

    Timeline (typical non-DST day in KST):
      08:50 ~ 15:40  →  KR session
      15:40 ~ 23:30  →  Idle (backup, retrain)
      23:30 ~ 06:00  →  US session
      06:00 ~ 08:50  →  Sleep
    """
    idle_done_today = False
    last_idle_date = None

    while not stop_event.is_set():
        now = datetime.now(KST)
        today = now.date()

        # ── KR market open? ──────────────────────────────────────
        if is_kr_market_open():
            idle_done_today = False
            try:
                await run_kr_session(config, stop_event)
            except Exception as e:
                logger.error("KR session error: %s", e)
            continue

        # ── US market open? ──────────────────────────────────────
        if is_us_market_open():
            try:
                await run_us_session(config, stop_event)
            except Exception as e:
                logger.error("US session error: %s", e)
            continue

        # ── Idle window ──────────────────────────────────────────
        if not idle_done_today and last_idle_date != today:
            try:
                await run_idle_tasks(config)
            except Exception as e:
                logger.warning("Idle tasks error: %s", e)
            idle_done_today = True
            last_idle_date = today

        # ── Determine next event and sleep ───────────────────────
        next_kr = next_kr_market_open()
        next_us = next_us_market_open()
        next_event = min(next_kr, next_us)
        wait_sec = _seconds_until(next_event)

        if wait_sec > 0:
            label = "KR" if next_event == next_kr else "US"
            logger.info(
                "Both markets closed. Next: %s at %s (%.0f min)",
                label,
                next_event.strftime("%Y-%m-%d %H:%M KST"),
                wait_sec / 60,
            )
            # Sleep in 5-min chunks so we can react to stop_event
            sleep_chunk = min(wait_sec, 300)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_chunk)
                break  # Stop event was set
            except asyncio.TimeoutError:
                continue


async def main() -> None:
    """
    Outer loop with auto-recovery.

    On unexpected crash: log the error, wait 30–60s, restart.
    On KeyboardInterrupt / SIGTERM: shut down cleanly.
    """
    setup_logging()

    config = Config()
    config.validate()
    config.log_summary()

    # Handle SIGTERM for clean shutdown in systemd / Docker
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Received shutdown signal.")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    while not stop_event.is_set():
        try:
            await scheduler(config, stop_event)
            break
        except KeyboardInterrupt:
            logger.info("Shutdown requested (Ctrl+C).")
            break
        except Exception as e:
            wait = random.uniform(30, 60)
            logger.error(
                "Fatal error: %s. Auto-restarting in %.0fs…", e, wait,
            )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=wait)
                break
            except asyncio.TimeoutError:
                continue

    logger.info("Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())
