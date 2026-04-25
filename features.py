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
    vol_index_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add all feature columns to an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume.
    index_df : pd.DataFrame, optional
        Market index (KOSPI / S&P 500) with a 'close' column.
    vol_index_df : pd.DataFrame, optional
        Volatility index (VKOSPI / VIX) with a 'close' column.

    Every feature is computed using only data up to and including time t.
    No negative shifts are used anywhere.
    """
    out = df.copy()

    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = _add_price_return_features(out)
    out = _add_technical_features(out)
    out = _add_volume_features(out)

    if index_df is not None:
        out = _add_market_context(out, index_df)

    out = _add_regime_features(out, index_df)

    if vol_index_df is not None and not vol_index_df.empty:
        vol_idx = vol_index_df["close"].reindex(out.index, method="ffill")
        out["vix_level"] = vol_idx
        out["vix_chg_5d"] = vol_idx / vol_idx.shift(5) - 1
        out["vix_percentile_252"] = vol_idx.rolling(252).rank(pct=True)
    else:
        out["vix_level"] = np.nan
        out["vix_chg_5d"] = np.nan
        out["vix_percentile_252"] = np.nan

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

    # Overnight gap: open_t / close_{t-1} - 1
    prev_close = close.shift(1)
    df["overnight_gap"] = df["open"] / prev_close - 1

    # Gap filled inside the session?
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    gap_up_filled = (df["open"] > prev_high) & (df["low"] <= prev_high)
    gap_dn_filled = (df["open"] < prev_low) & (df["high"] >= prev_low)
    df["gap_filled"] = (gap_up_filled | gap_dn_filled).astype(int)

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

    # Volume imbalance — Lee-Ready-style approximation:
    # Up days count as buy-initiated, down days as sell-initiated.
    direction = np.where(df["close"] > df["open"], 1, -1)
    signed_vol = direction * vol
    buy_vol_20 = pd.Series(
        np.where(signed_vol > 0, signed_vol, 0),
        index=df.index,
    ).rolling(20).sum()
    total_vol_20 = vol.rolling(20).sum()
    df["vol_imbalance_20"] = np.where(
        total_vol_20 > 0,
        buy_vol_20 / total_vol_20,
        0.5,
    )
    df["vol_imbalance_trend"] = (
        pd.Series(df["vol_imbalance_20"], index=df.index)
        - pd.Series(df["vol_imbalance_20"], index=df.index).shift(5)
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
# 5. Market Regime Features
# ═══════════════════════════════════════════════════════════════════════

def _add_regime_features(
    df: pd.DataFrame,
    index_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Market regime indicators: volatility regime, trend strength/direction,
    and (optionally) trend alignment with a market index.
    """
    # Volatility regime: current 20-day vol / 1-year vol
    vol_20 = df["log_ret_1d"].rolling(20).std()
    vol_252 = df["log_ret_1d"].rolling(252).std()
    df["vol_regime"] = np.where(vol_252 > 0, vol_20 / vol_252, 1.0)

    # Trend strength: |EMA20 - EMA60| / ATR (unit-free)
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema60 = df["close"].ewm(span=60, adjust=False).mean()
    atr14 = (
        df["atr_norm"] * df["close"] if "atr_norm" in df.columns
        else df["close"] * 0.02
    )
    df["trend_strength"] = (ema20 - ema60).abs() / atr14.replace(0, np.nan)

    diff = ema20 - ema60
    df["trend_direction"] = np.sign(diff).fillna(0)

    if index_df is not None and not index_df.empty:
        idx_close = index_df["close"].reindex(df.index, method="ffill")
        idx_ema20 = idx_close.ewm(span=20, adjust=False).mean()
        idx_ema60 = idx_close.ewm(span=60, adjust=False).mean()
        df["index_trend_direction"] = np.sign(idx_ema20 - idx_ema60).fillna(0)
        df["trend_alignment"] = (
            df["trend_direction"] * df["index_trend_direction"]
        )
    else:
        df["index_trend_direction"] = 0
        df["trend_alignment"] = 0

    return df


# ═══════════════════════════════════════════════════════════════════════
# Utility: list feature column names
# ═══════════════════════════════════════════════════════════════════════

OHLCV_COLS = {"open", "high", "low", "close", "volume"}


def feature_columns(
    include_market: bool = False,
    include_regime: bool = True,
    include_vol_index: bool = False,
) -> list[str]:
    """Return the list of feature column names (excluding OHLCV)."""
    cols = [
        # Price / Return
        "log_ret_1d", "log_ret_5d", "log_ret_20d", "log_ret_60d",
        "volatility_20d", "volatility_60d",
        "intraday_range", "close_location",
        "overnight_gap", "gap_filled",
        # Technical
        "rsi_14", "macd", "macd_signal", "macd_hist", "bb_pctb",
        "sma_dev_5", "sma_dev_20", "sma_dev_60",
        "atr_norm",
        # Volume
        "vol_chg_1d", "vol_chg_5d", "obv_roc_20", "turnover_ratio_20",
        "vol_imbalance_20", "vol_imbalance_trend",
    ]
    if include_market:
        cols += ["excess_ret_5d", "excess_ret_20d", "beta_60d"]
    if include_regime:
        cols += [
            "vol_regime", "trend_strength", "trend_direction",
            "index_trend_direction", "trend_alignment",
        ]
    if include_vol_index:
        cols += ["vix_level", "vix_chg_5d", "vix_percentile_252"]
    return cols
