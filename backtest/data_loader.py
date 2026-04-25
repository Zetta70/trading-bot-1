"""
Historical Data Loader.

Supports both Korean and US equities:
  - KR stocks: FinanceDataReader (6-digit codes like '005930')
  - US stocks: yfinance (alphabetic tickers like 'AAPL', 'NVDA')

Ticker format convention:
  - 'KR:005930' or plain '005930' → Korean stock
  - 'US:AAPL'  or plain 'AAPL'   → US stock
  - Auto-detection: all-digit = KR, otherwise = US

NOTE: Survivorship bias caveat — only currently listed tickers are available
through FinanceDataReader / yfinance. Delisted stocks are not included, which
may overstate backtest performance. This is a known limitation.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Ticker Parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_ticker(raw: str) -> tuple[str, str]:
    """
    Parse a ticker string into (market, symbol).

    Returns
    -------
    tuple[str, str]
        ('KR', '005930') or ('US', 'AAPL')
    """
    raw = raw.strip()
    if raw.upper().startswith("KR:"):
        return ("KR", raw[3:].strip())
    if raw.upper().startswith("US:"):
        return ("US", raw[3:].strip())
    # Auto-detect: all digits → KR, else → US
    if re.fullmatch(r"\d{6}", raw):
        return ("KR", raw)
    return ("US", raw.upper())


# ═══════════════════════════════════════════════════════════════════════
# Unified Loader
# ═══════════════════════════════════════════════════════════════════════

def load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load daily OHLCV data for any ticker.

    Automatically routes to the correct data source based on
    ticker format (KR: or US: prefix, or auto-detect).

    Parameters
    ----------
    ticker : str
        Ticker with optional prefix: 'KR:005930', 'US:AAPL', '005930', 'AAPL'.
    start, end : str
        Date range in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume.
        Index: DatetimeIndex.
    """
    market, symbol = parse_ticker(ticker)

    if market == "KR":
        return _load_kr_ohlcv(symbol, start, end)
    else:
        return _load_us_ohlcv(symbol, start, end)


def load_kospi(start: str, end: str) -> pd.DataFrame:
    """Load daily OHLCV for the KOSPI index."""
    return _load_kr_index("KS11", "KOSPI", start, end)


def load_sp500(start: str, end: str) -> pd.DataFrame:
    """Load daily OHLCV for the S&P 500 index."""
    return _load_us_ohlcv("^GSPC", start, end, label="S&P500")


def load_vkospi(start: str, end: str) -> pd.DataFrame:
    """
    Load VKOSPI (KOSPI implied volatility index).

    VKOSPI > 25 signals market stress; < 15 signals complacency.
    Returns empty DataFrame if the data source can't serve the range.
    """
    import FinanceDataReader as fdr
    logger.info("Loading VKOSPI (%s ~ %s)", start, end)
    try:
        df = fdr.DataReader("VKOSPI", start, end)
        df.columns = [c.lower() for c in df.columns]
        df = df[["close"]].dropna()
        logger.info("Loaded %d VKOSPI rows", len(df))
        return df
    except Exception as e:
        logger.warning("Failed to load VKOSPI: %s", e)
        return pd.DataFrame(columns=["close"])


def load_vix(start: str, end: str) -> pd.DataFrame:
    """Load CBOE VIX (for US stocks regime features)."""
    logger.info("Loading VIX (%s ~ %s)", start, end)
    try:
        import yfinance as yf
        df = yf.download("^VIX", start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.columns = [c.lower() for c in df.columns]
        df = df[["close"]].dropna()
        logger.info("Loaded %d VIX rows", len(df))
        return df
    except Exception as e:
        logger.warning("Failed to load VIX: %s", e)
        return pd.DataFrame(columns=["close"])


def load_usdkrw(start: str, end: str) -> pd.DataFrame:
    """
    Load daily USD/KRW exchange rate.

    Returns
    -------
    pd.DataFrame
        Columns: close (KRW per 1 USD). Index: DatetimeIndex.
    """
    logger.info("Loading USD/KRW FX rate (%s ~ %s)", start, end)
    try:
        import yfinance as yf
        df = yf.download("USDKRW=X", start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.columns = [c.lower() for c in df.columns]
        df = df[["close"]].dropna()
        logger.info("Loaded %d FX rows", len(df))
        return df
    except Exception as e:
        logger.warning("Failed to load USD/KRW: %s", e)
        return pd.DataFrame(columns=["close"])


# ═══════════════════════════════════════════════════════════════════════
# KR: FinanceDataReader
# ═══════════════════════════════════════════════════════════════════════

def _load_kr_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load Korean stock OHLCV via FinanceDataReader."""
    import FinanceDataReader as fdr

    logger.info("Loading KR OHLCV for %s (%s ~ %s)", ticker, start, end)
    df = fdr.DataReader(ticker, start, end)

    df.columns = [c.lower() for c in df.columns]

    required = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"KR:{ticker}: missing columns {missing_cols}. "
            f"Available: {list(df.columns)}"
        )
    df = df[required].copy()
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    logger.info("Loaded %d rows for KR:%s", len(df), ticker)
    return df


def _load_kr_index(code: str, label: str, start: str, end: str) -> pd.DataFrame:
    """Load a KR index via FinanceDataReader."""
    import FinanceDataReader as fdr

    logger.info("Loading %s index (%s ~ %s)", label, start, end)
    df = fdr.DataReader(code, start, end)
    df.columns = [c.lower() for c in df.columns]

    required = ["open", "high", "low", "close", "volume"]
    available = [c for c in required if c in df.columns]
    df = df[available].copy()

    for col in required:
        if col not in df.columns:
            df[col] = 0

    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    logger.info("Loaded %d rows for %s", len(df), label)
    return df


# ═══════════════════════════════════════════════════════════════════════
# US: yfinance
# ═══════════════════════════════════════════════════════════════════════

def _load_us_ohlcv(
    ticker: str, start: str, end: str, label: str | None = None,
) -> pd.DataFrame:
    """Load US stock/index OHLCV via yfinance."""
    import yfinance as yf

    display = label or f"US:{ticker}"
    logger.info("Loading %s OHLCV (%s ~ %s)", display, start, end)

    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"{display}: yfinance returned no data")

    # yfinance may return MultiIndex columns for single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.columns = [c.lower() for c in df.columns]

    # Map yfinance columns: 'adj close' → drop, keep standard OHLCV
    required = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{display}: missing columns {missing_cols}. "
            f"Available: {list(df.columns)}"
        )

    df = df[required].copy()
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    # Ensure index is timezone-naive DatetimeIndex (match KR format)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    logger.info("Loaded %d rows for %s", len(df), display)
    return df
