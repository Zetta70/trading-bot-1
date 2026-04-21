"""
Central configuration loaded from .env and environment variables.

All constants, market hours, and timezone settings live here.
Supports both Korean (KR) and US stock markets.
"""

import os
import sys
import logging
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ── Timezones ───────────────────────────────────────────────────────
KST = ZoneInfo("Asia/Seoul")
EST = ZoneInfo("America/New_York")

# ── Market Hours ────────────────────────────────────────────────────
# Korean market (KST)
MARKET_OPEN = (8, 50)     # 08:50 KST
MARKET_CLOSE = (15, 40)   # 15:40 KST

# US market (ET — automatically handles EST/EDT via zoneinfo)
US_MARKET_OPEN = (9, 30)   # 09:30 ET
US_MARKET_CLOSE = (16, 0)  # 16:00 ET


class Config:
    """Immutable configuration snapshot from environment."""

    def __init__(self):
        load_dotenv()

        # Run mode
        self.run_mode: str = os.getenv("RUN_MODE", "MOCK").upper()

        # ── KIS API credentials (shared or KR-specific) ─────────
        self.api_key: str = os.getenv("KIS_API_KEY", "")
        self.api_secret: str = os.getenv("KIS_API_SECRET", "")
        self.acc_no: str = os.getenv("KIS_ACC_NO", "")

        # ── KIS US API credentials (optional, falls back to shared) ─
        self.us_api_key: str = os.getenv("KIS_US_API_KEY", "") or self.api_key
        self.us_api_secret: str = os.getenv("KIS_US_API_SECRET", "") or self.api_secret
        self.us_acc_no: str = os.getenv("KIS_US_ACC_NO", "") or self.acc_no

        # ── Tickers ─────────────────────────────────────────────
        # Legacy: TICKERS (backward compat → treated as KR)
        raw_legacy = os.getenv("TICKERS", "")

        raw_kr = os.getenv("TICKERS_KR", "")
        raw_us = os.getenv("TICKERS_US", "")

        # KR tickers: TICKERS_KR takes priority, fallback to TICKERS
        kr_source = raw_kr or raw_legacy or "005930,000660,035420"
        self.tickers_kr: list[str] = [
            t.strip() for t in kr_source.split(",") if t.strip()
        ]
        # US tickers
        self.tickers_us: list[str] = [
            t.strip() for t in raw_us.split(",") if t.strip()
        ] if raw_us else []

        # Combined for backward compat (existing code reads self.tickers)
        self.tickers: list[str] = self.tickers_kr.copy()

        # ── Portfolio ───────────────────────────────────────────
        self.initial_cash: int = int(os.getenv("INITIAL_CASH", "10000000"))
        self.max_drawdown_pct: float = float(
            os.getenv("MAX_DRAWDOWN_PCT", "0.03")
        )
        self.max_active_bots: int = int(os.getenv("MAX_ACTIVE_BOTS", "10"))

        # US allocation (fraction of initial_cash for US market)
        self.us_allocation_pct: float = float(
            os.getenv("US_ALLOCATION_PCT", "0.3")
        )

        # ── Strategy ────────────────────────────────────────────
        self.poll_interval: int = int(os.getenv("POLL_INTERVAL", "10"))
        self.sma_period: int = int(os.getenv("SMA_PERIOD", "20"))
        self.threshold: float = float(os.getenv("THRESHOLD", "0.005"))
        self.bullish_threshold: float = float(
            os.getenv("BULLISH_THRESHOLD", "0.75")
        )
        self.trailing_stop_pct: float = float(
            os.getenv("TRAILING_STOP_PCT", "0.02")
        )

        # ── Scanner ─────────────────────────────────────────────
        self.scan_interval: int = int(os.getenv("SCAN_INTERVAL", "900"))

        # ── ML / Backtest ───────────────────────────────────────
        self.ml_model_path: str = os.getenv("ML_MODEL_PATH", "models/lgbm_v1.pkl")
        self.ml_threshold_buy: float = float(os.getenv("ML_THRESHOLD_BUY", "0.55"))
        self.ml_threshold_sell: float = float(os.getenv("ML_THRESHOLD_SELL", "0.45"))
        self.bt_train_window: int = int(os.getenv("BT_TRAIN_WINDOW", "756"))
        self.bt_test_window: int = int(os.getenv("BT_TEST_WINDOW", "63"))
        self.bt_horizon: int = int(os.getenv("BT_HORIZON", "5"))
        self.bt_target_threshold: float = float(os.getenv("BT_TARGET_THRESHOLD", "0.02"))
        self.stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "-0.05"))

        # ── FX (Exchange Rate) ──────────────────────────────────
        # Default USD/KRW rate used when real-time fetch fails
        self.default_usdkrw: float = float(os.getenv("DEFAULT_USDKRW", "1380.0"))

    def validate(self) -> None:
        """Exit with error if KIS credentials are missing."""
        missing = []
        if not self.api_key:
            missing.append("KIS_API_KEY")
        if not self.api_secret:
            missing.append("KIS_API_SECRET")
        if not self.acc_no:
            missing.append("KIS_ACC_NO")
        if missing:
            logger.error(
                "Missing required credentials: %s. "
                "Fill in .env (see .env.example).",
                ", ".join(missing),
            )
            sys.exit(1)

        if not self.tickers_kr and not self.tickers_us:
            logger.error("At least one of TICKERS_KR or TICKERS_US must be set.")
            sys.exit(1)

    def log_summary(self) -> None:
        """Log the active configuration."""
        logger.info("=" * 60)
        logger.info("Configuration")
        logger.info("  Run Mode:       %s", self.run_mode)
        logger.info("  KR Tickers:     %s", ", ".join(self.tickers_kr) or "(none)")
        logger.info("  US Tickers:     %s", ", ".join(self.tickers_us) or "(none)")
        logger.info("  Initial Cash:   %s KRW", f"{self.initial_cash:,}")
        logger.info("  US Allocation:  %.0f%%", self.us_allocation_pct * 100)
        logger.info("  Max Drawdown:   %.1f%%", self.max_drawdown_pct * 100)
        logger.info("  Poll Interval:  %ds", self.poll_interval)
        logger.info("  SMA Period:     %d", self.sma_period)
        logger.info("  Threshold:      ±%.2f%%", self.threshold * 100)
        logger.info("  ML Bullish:     ≥%.2f", self.bullish_threshold)
        logger.info("  Trailing Stop:  %.1f%%", self.trailing_stop_pct * 100)
        logger.info("  Scan Interval:  %ds", self.scan_interval)
        logger.info("  Max Bots:       %d", self.max_active_bots)
        logger.info("  ML Model:       %s", self.ml_model_path)
        logger.info("  ML Buy/Sell:    ≥%.2f / ≤%.2f",
                     self.ml_threshold_buy, self.ml_threshold_sell)
        logger.info("  Default USD/KRW: %.1f", self.default_usdkrw)
        logger.info("=" * 60)
