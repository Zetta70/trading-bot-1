"""
Position Sizing — Kelly, Volatility Targeting, and Correlation Penalty.

Three sizing methodologies combined:
  1. Fractional Kelly: size scales with ML edge (prob - 0.5)
  2. Volatility targeting: normalize each position to contribute equal risk
  3. Correlation penalty: reduce size when similar positions already held

Final position size = min(kelly, vol) * correlation_multiplier * equity.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def kelly_fraction(
    win_prob: float,
    avg_win: float = 2.0,
    avg_loss: float = 1.0,
    kelly_multiplier: float = 0.25,
) -> float:
    """
    Fractional Kelly sizing.

    Full Kelly is theoretically optimal but assumes known probabilities.
    ML probabilities are noisy estimates, so full Kelly dramatically
    overbets. Fractional Kelly (25% of full) is the industry standard.

    f* = (b*p - q) / b where b = avg_win/avg_loss, p = win_prob, q = 1-p
    """
    if win_prob <= 0.5:
        return 0.0
    b = avg_win / max(avg_loss, 1e-6)
    p = win_prob
    q = 1 - p
    full_kelly = (b * p - q) / b
    if full_kelly <= 0:
        return 0.0
    return float(kelly_multiplier * full_kelly)


def vol_targeted_weight(
    stock_annualized_vol: float,
    target_portfolio_vol: float = 0.15,
    max_weight: float = 0.10,
) -> float:
    """
    Volatility-targeted position weight.

    Allocates inverse-proportionally to a stock's own volatility so that
    each position contributes roughly equal portfolio variance.
    """
    if stock_annualized_vol <= 0 or np.isnan(stock_annualized_vol):
        return 0.0
    raw = target_portfolio_vol / stock_annualized_vol
    return float(min(raw, max_weight))


def correlation_penalty(
    existing_returns: dict[str, pd.Series],
    candidate_returns: pd.Series,
    high_corr_threshold: float = 0.70,
    lookback: int = 60,
) -> float:
    """
    Returns multiplier in [0, 1]:
      - 0 correlated positions -> 1.0
      - 1 correlated           -> 0.5
      - 2 correlated           -> 0.25
      - 3+ correlated          -> 0.0 (reject)
    """
    if not existing_returns:
        return 1.0

    cand_tail = candidate_returns.dropna().tail(lookback)
    if len(cand_tail) < lookback // 2:
        return 1.0

    high_corr_count = 0
    for ticker, ret in existing_returns.items():
        ret_tail = ret.dropna().tail(lookback)
        aligned = pd.concat([cand_tail, ret_tail], axis=1).dropna()
        if len(aligned) < lookback // 2:
            continue
        c = aligned.corr().iloc[0, 1]
        if pd.notna(c) and abs(c) >= high_corr_threshold:
            high_corr_count += 1

    if high_corr_count == 0:
        return 1.0
    elif high_corr_count == 1:
        return 0.5
    elif high_corr_count == 2:
        return 0.25
    return 0.0


def compute_position_size(
    portfolio_equity: float,
    ml_probability: float,
    stock_annualized_vol: float,
    candidate_returns: pd.Series,
    existing_returns: dict[str, pd.Series],
    *,
    kelly_multiplier: float = 0.25,
    target_portfolio_vol: float = 0.15,
    max_weight: float = 0.10,
    pt_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
) -> tuple[float, dict]:
    """
    Combine all three sizing methods: take the min of Kelly and vol target,
    then apply the correlation penalty as a multiplier.

    Returns (krw_amount, debug_dict).
    """
    kelly_w = kelly_fraction(
        win_prob=ml_probability,
        avg_win=pt_atr_mult,
        avg_loss=sl_atr_mult,
        kelly_multiplier=kelly_multiplier,
    )
    vol_w = vol_targeted_weight(
        stock_annualized_vol=stock_annualized_vol,
        target_portfolio_vol=target_portfolio_vol,
        max_weight=max_weight,
    )
    corr_mult = correlation_penalty(
        existing_returns=existing_returns,
        candidate_returns=candidate_returns,
    )

    final_weight = min(kelly_w, vol_w) * corr_mult
    final_weight = max(0.0, min(final_weight, max_weight))
    amount = portfolio_equity * final_weight

    debug = {
        "kelly_w": round(kelly_w, 4),
        "vol_w": round(vol_w, 4),
        "corr_mult": round(corr_mult, 4),
        "final_w": round(final_weight, 4),
        "amount": round(amount, 0),
    }
    return amount, debug
