"""
Feature Engineering Module.

Computes technical features from OHLCV data for ML-based stock prediction.
All features use only data available at time t (no look-ahead bias).

Categories:
  1. Price / Return features (log returns, volatility, intraday range)
  2. Technical indicators (RSI, MACD, Bollinger, SMA deviation, ATR)
  3. Volume features (volume change, OBV, turnover ratio)
  4. Market context (excess return vs index, rolling beta)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def add_features(
    df: pd.DataFrame,
    index_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add all feature columns to an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume.
        Index should be datetime (KST).
    index_df : pd.DataFrame, optional
        KOSPI index OHLCV with at least a 'close' column and matching
        datetime index. Used for market-context features.

    Returns
    -------
    pd.DataFrame
        Original columns + all computed features.
        NaN rows are preserved (caller decides how to handle them).

    Notes
    -----
    Every feature is computed using only data up to and including time t.
    No negative shifts (e.g. shift(-1)) are used anywhere.
    """
    out = df.copy()

    # Ensure numeric types
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # ── 1. Price / Return ────────────────────────────────────────
    out = _add_price_return_features(out)

    # ── 2. Technical Indicators ──────────────────────────────────
    out = _add_technical_features(out)

    # ── 3. Volume ────────────────────────────────────────────────
    out = _add_volume_features(out)

    # ── 4. Market Context ────────────────────────────────────────
    if index_df is not None:
        out = _add_market_context(out, index_df)

    return out


# ═══════════════════════════════════════════════════════════════════════
# 1. Price / Return Features
# ═══════════════════════════════════════════════════════════════════════

def _add_price_return_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    log_close = np.log(close)

    # Log returns for various horizons
    for n in (1, 5, 20, 60):
        df[f"log_ret_{n}d"] = log_close - log_close.shift(n)

    # Rolling volatility (std of 1-day log returns)
    log_ret_1d = df["log_ret_1d"]
    for n in (20, 60):
        df[f"volatility_{n}d"] = log_ret_1d.rolling(n).std()

    # Intraday range: (high - low) / close
    df["intraday_range"] = (df["high"] - df["low"]) / close

    # Close location value: (close - low) / (high - low)
    hl_range = df["high"] - df["low"]
    df["close_location"] = np.where(
        hl_range > 0,
        (close - df["low"]) / hl_range,
        0.5,  # flat candle → neutral
    )

    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. Technical Indicators
# ═══════════════════════════════════════════════════════════════════════

def _add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    # ── RSI(14) ──────────────────────────────────────────────────
    df["rsi_14"] = _rsi(close, 14)

    # ── MACD(12, 26, 9) ─────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line

    # ── Bollinger Bands (20, 2σ) → %B ────────────────────────────
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    band_width = upper - lower
    df["bb_pctb"] = np.where(
        band_width > 0,
        (close - lower) / band_width,
        0.5,
    )

    # ── SMA deviation: close / SMA(n) - 1 ───────────────────────
    for n in (5, 20, 60):
        sma_n = close.rolling(n).mean()
        df[f"sma_dev_{n}"] = close / sma_n - 1

    # ── ATR(14) normalized ───────────────────────────────────────
    high, low = df["high"], df["low"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df["atr_norm"] = atr14 / close

    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI using exponential moving average of gains/losses."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# ═══════════════════════════════════════════════════════════════════════
# 3. Volume Features
# ═══════════════════════════════════════════════════════════════════════

def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    vol = df["volume"].astype(float)

    # Volume change rate
    for n in (1, 5):
        prev = vol.shift(n)
        df[f"vol_chg_{n}d"] = np.where(prev > 0, vol / prev - 1, 0.0)

    # OBV 20-day rate of change
    sign = np.sign(df["close"].diff())
    obv = (sign * vol).cumsum()
    obv_20 = obv.shift(20)
    df["obv_roc_20"] = np.where(
        obv_20.abs() > 0,
        (obv - obv_20) / obv_20.abs(),
        0.0,
    )

    # Turnover (volume * close) vs 20-day MA
    turnover = vol * df["close"]
    turnover_ma20 = turnover.rolling(20).mean()
    df["turnover_ratio_20"] = np.where(
        turnover_ma20 > 0,
        turnover / turnover_ma20,
        1.0,
    )

    return df


# ═══════════════════════════════════════════════════════════════════════
# 4. Market Context Features
# ═══════════════════════════════════════════════════════════════════════

def _add_market_context(
    df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add features that compare the stock to a market index."""
    idx_close = index_df["close"].reindex(df.index, method="ffill")
    idx_log_ret = np.log(idx_close) - np.log(idx_close.shift(1))
    stock_log_ret = df["log_ret_1d"]

    # Excess return over index
    for n in (5, 20):
        stock_cum = stock_log_ret.rolling(n).sum()
        idx_cum = idx_log_ret.rolling(n).sum()
        df[f"excess_ret_{n}d"] = stock_cum - idx_cum

    # 60-day rolling beta
    cov = stock_log_ret.rolling(60).cov(idx_log_ret)
    var = idx_log_ret.rolling(60).var()
    df["beta_60d"] = np.where(var > 0, cov / var, 1.0)

    return df


# ═══════════════════════════════════════════════════════════════════════
# Utility: list feature column names
# ═══════════════════════════════════════════════════════════════════════

OHLCV_COLS = {"open", "high", "low", "close", "volume"}


def feature_columns(include_market: bool = False) -> list[str]:
    """Return the list of feature column names (excluding OHLCV)."""
    cols = [
        # Price / Return
        "log_ret_1d", "log_ret_5d", "log_ret_20d", "log_ret_60d",
        "volatility_20d", "volatility_60d",
        "intraday_range", "close_location",
        # Technical
        "rsi_14", "macd", "macd_signal", "macd_hist", "bb_pctb",
        "sma_dev_5", "sma_dev_20", "sma_dev_60",
        "atr_norm",
        # Volume
        "vol_chg_1d", "vol_chg_5d", "obv_roc_20", "turnover_ratio_20",
    ]
    if include_market:
        cols += ["excess_ret_5d", "excess_ret_20d", "beta_60d"]
    return cols
