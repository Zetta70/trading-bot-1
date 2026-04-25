"""
Daily OHLCV Cache for Live ML Inference.

Provides a thin wrapper around backtest.data_loader.load_ohlcv with
in-memory caching and TTL. Used by the live bot/scanner to evaluate
ML signals on daily bars rather than live ticks.

Why not use tick data?
  - Training features (RSI_14, volatility_60d, SMA_dev, etc.) are
    defined on *daily* returns. Feeding 10-second tick data through
    the same feature pipeline produces near-zero volatility and
    RSI stuck near 50 — destroying the ML signal.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from threading import Lock

import pandas as pd

from config import KST

logger = logging.getLogger(__name__)


class OHLCVCache:
    """Thread-safe daily OHLCV cache with TTL (default 1 hour)."""

    def __init__(self, ttl_seconds: int = 3600, lookback_days: int = 180):
        self.ttl = ttl_seconds
        self.lookback_days = lookback_days
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._lock = Lock()

    def get(self, ticker: str) -> pd.DataFrame | None:
        """Return cached or freshly loaded daily OHLCV. None on failure."""
        now = time.time()
        with self._lock:
            if ticker in self._cache:
                ts, df = self._cache[ticker]
                if now - ts < self.ttl:
                    return df

        try:
            from backtest.data_loader import load_ohlcv
            end = datetime.now(KST).strftime("%Y-%m-%d")
            start = (
                datetime.now(KST) - timedelta(days=self.lookback_days)
            ).strftime("%Y-%m-%d")
            df = load_ohlcv(ticker, start, end)

            if df is None or len(df) < 60:
                logger.warning(
                    "OHLCV for %s too short (%d rows) — skipping cache.",
                    ticker, 0 if df is None else len(df),
                )
                return None

            with self._lock:
                self._cache[ticker] = (now, df)
            return df

        except Exception as e:
            logger.warning("Failed to load OHLCV for %s: %s", ticker, e)
            return None

    def invalidate(self, ticker: str | None = None) -> None:
        """Clear cache for a single ticker or all tickers."""
        with self._lock:
            if ticker is None:
                self._cache.clear()
            else:
                self._cache.pop(ticker, None)


_GLOBAL_CACHE: OHLCVCache | None = None


def get_cache() -> OHLCVCache:
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = OHLCVCache()
    return _GLOBAL_CACHE
