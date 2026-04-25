"""
Monte Carlo Bootstrap for Strategy Robustness Testing.

Two tests:
  1. Trade Shuffling — randomize the order of observed trade returns;
     tells you whether compounding order matters or your edge is just
     "good average trades".
  2. Block Bootstrap — resample blocks of daily returns under the null
     hypothesis of zero mean (preserves short-term autocorrelation);
     tells you whether the realized Sharpe exceeds the bootstrap null
     at 95%/99% confidence.

If the real backtest result is deep in the tail of the bootstrap
distribution, the edge is statistically significant.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backtest.metrics import sharpe_ratio

logger = logging.getLogger(__name__)


def trade_shuffle_mc(
    trade_returns: pd.Series,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Shuffle the order of trade returns, compute total return + max DD on
    each permutation. The percentile of the real result vs the shuffled
    distribution tells you whether *order* matters (timing edge) vs the
    overall trade quality.
    """
    rng = np.random.default_rng(seed)
    trades = trade_returns.dropna().values
    n = len(trades)
    if n < 20:
        return {"error": "insufficient trades"}

    real_equity = (1 + trades).cumprod()
    real_return = float(real_equity[-1] - 1)
    real_dd = float(
        (real_equity / np.maximum.accumulate(real_equity) - 1).min()
    )

    mc_returns = []
    mc_dds = []
    for _ in range(n_simulations):
        shuffled = rng.permutation(trades)
        eq = (1 + shuffled).cumprod()
        mc_returns.append(eq[-1] - 1)
        mc_dds.append(float((eq / np.maximum.accumulate(eq) - 1).min()))

    mc_returns_arr = np.array(mc_returns)
    mc_dds_arr = np.array(mc_dds)

    return {
        "real_total_return": real_return,
        "mc_return_mean": float(mc_returns_arr.mean()),
        "mc_return_std": float(mc_returns_arr.std()),
        "return_percentile": float((mc_returns_arr <= real_return).mean()),
        "real_max_dd": real_dd,
        "mc_dd_mean": float(mc_dds_arr.mean()),
        "dd_percentile": float((mc_dds_arr >= real_dd).mean()),
    }


def block_bootstrap_mc(
    daily_returns: pd.Series,
    n_simulations: int = 1000,
    block_size: int = 20,
    seed: int = 42,
) -> dict:
    """
    Block bootstrap of daily returns under a zero-mean null. The blocks
    preserve short-term autocorrelation that an iid bootstrap would
    destroy. The Sharpe of each bootstrap path is compared to the real
    Sharpe — the empirical p-value is the share of bootstrap Sharpes
    ≥ the real one.
    """
    rng = np.random.default_rng(seed)
    rets = daily_returns.dropna().values
    n = len(rets)
    if n < 100:
        return {"error": "insufficient returns"}

    real_sharpe = sharpe_ratio(pd.Series(rets))

    null_rets = rets - rets.mean()  # zero-mean null

    mc_sharpes = []
    for _ in range(n_simulations):
        bootstrap: list = []
        while len(bootstrap) < n:
            start = rng.integers(0, max(1, n - block_size))
            bootstrap.extend(null_rets[start:start + block_size])
        bootstrap_arr = np.array(bootstrap[:n])
        mc_sharpes.append(sharpe_ratio(pd.Series(bootstrap_arr)))

    mc_sharpes_arr = np.array(mc_sharpes)
    p_value = float((mc_sharpes_arr >= real_sharpe).mean())

    return {
        "real_sharpe": real_sharpe,
        "mc_sharpe_mean": float(mc_sharpes_arr.mean()),
        "mc_sharpe_std": float(mc_sharpes_arr.std()),
        "mc_sharpe_95pct": float(np.percentile(mc_sharpes_arr, 95)),
        "mc_sharpe_99pct": float(np.percentile(mc_sharpes_arr, 99)),
        "p_value": p_value,
        "is_significant_95": p_value < 0.05,
        "is_significant_99": p_value < 0.01,
    }
