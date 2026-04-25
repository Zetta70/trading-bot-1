"""
Core Technical Indicators for Adaptive Trading.

Unlike the static thresholds used in legacy code (±0.5% momentum),
these indicators adapt to each stock's own volatility regime.

Key functions:
  - atr(df, period=14)              : Average True Range (Wilder)
  - donchian_channel(df, period=20) : Rolling high/low bands (shift(1))
  - realized_volatility(df, period) : Annualized return std
  - adaptive_breakout_level         : Entry threshold
  - chandelier_stop                 : Dynamic trailing stop
  - triple_barrier_labels           : Path-aware target labels
  - binary_pt_labels                : Binary version for LightGBM
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range — Wilder's smoothing.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def donchian_channel(
    df: pd.DataFrame,
    period: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    Rolling Donchian high / low channel.

    Uses shift(1) so the channel at time t does NOT include time t's bar —
    prevents lookahead bias when used as a breakout signal.
    """
    upper = df["high"].rolling(period).max().shift(1)
    lower = df["low"].rolling(period).min().shift(1)
    return upper, lower


def realized_volatility(
    df: pd.DataFrame,
    period: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """Log-return rolling standard deviation, optionally annualized."""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol = log_ret.rolling(period).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def adaptive_breakout_level(
    df: pd.DataFrame,
    atr_mult: float = 1.5,
    atr_period: int = 14,
) -> pd.Series:
    """
    Price level above which a breakout is statistically significant.

    Level at time t = close_t + atr_mult * ATR_t.
    """
    return df["close"] + atr_mult * atr(df, atr_period)


def chandelier_stop(
    highest_price: float,
    current_atr: float,
    atr_mult: float = 3.0,
) -> float:
    """Dynamic trailing stop price: highest_since_entry - atr_mult * ATR."""
    return highest_price - atr_mult * current_atr


def triple_barrier_labels(
    df: pd.DataFrame,
    pt_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    max_hold: int = 10,
    atr_period: int = 14,
) -> pd.Series:
    """
    Triple Barrier Method labelling (Lopez de Prado).

    For each bar t, simulate entering a long position and check which
    barrier is hit first within max_hold days:
      - Profit target: entry + pt_atr_mult * ATR_t → label +1
      - Stop loss:     entry - sl_atr_mult * ATR_t → label -1
      - Time expiry:   max_hold bars elapse        → label  0

    Conservative tie-breaker: if both barriers hit on the same bar,
    assume SL was touched first (we cannot see intrabar order).

    Returns pd.Series of {+1, 0, -1}; NaN at the tail where future is unknown.
    """
    atr_series = atr(df, atr_period)
    labels = pd.Series(index=df.index, dtype="float64")

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = atr_series.values
    n = len(df)

    for t in range(n - 1):
        a = atrs[t]
        if np.isnan(a) or a <= 0:
            continue
        entry = closes[t]
        pt = entry + pt_atr_mult * a
        sl = entry - sl_atr_mult * a
        label = 0
        for k in range(1, max_hold + 1):
            idx = t + k
            if idx >= n:
                label = np.nan
                break
            if lows[idx] <= sl:
                label = -1
                break
            if highs[idx] >= pt:
                label = 1
                break
        labels.iloc[t] = label

    return labels


def binary_pt_labels(
    df: pd.DataFrame,
    pt_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    max_hold: int = 10,
    atr_period: int = 14,
) -> pd.Series:
    """
    Binary version of triple barrier: 1 if profit target hit, else 0.

    Use when training a binary classifier with LightGBM's
    `objective='binary'`.
    """
    tri = triple_barrier_labels(
        df, pt_atr_mult, sl_atr_mult, max_hold, atr_period,
    )
    binary = (tri == 1).astype("Int64")
    binary[tri.isna()] = pd.NA
    return binary
