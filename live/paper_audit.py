"""
Live Performance Audit — detect backtest/paper-trading drift.

Compares live trade-return distribution to the backtest distribution
via the two-sample Kolmogorov-Smirnov test. If KS rejects same-
distribution at p < 0.05, halt trading and investigate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def audit_live_vs_backtest(
    live_trade_log: str,
    backtest_trade_log: str,
    min_live_trades: int = 20,
) -> dict:
    """
    KS test of live vs backtest trade-return distributions.

    Returns
    -------
    dict with ks_statistic, p_value, drift_detected (bool), and a
    recommendation string. ``status="insufficient_data"`` if too few
    live trades to test yet.
    """
    if not Path(live_trade_log).exists():
        return {"error": "live log not found"}
    if not Path(backtest_trade_log).exists():
        return {"error": "backtest log not found"}

    live = pd.read_csv(live_trade_log)
    bt = pd.read_csv(backtest_trade_log)

    if "return_pct" not in live.columns:
        return {"error": "live log lacks return_pct; cannot audit"}

    live_rets = live["return_pct"].dropna()
    bt_rets = bt["return_pct"].dropna()

    if len(live_rets) < min_live_trades:
        return {
            "status": "insufficient_data",
            "live_trades": int(len(live_rets)),
            "min_required": min_live_trades,
        }

    ks_stat, p_value = stats.ks_2samp(live_rets, bt_rets)
    drift = bool(p_value < 0.05)

    result = {
        "live_mean": float(live_rets.mean()),
        "bt_mean": float(bt_rets.mean()),
        "live_std": float(live_rets.std()),
        "bt_std": float(bt_rets.std()),
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "drift_detected": drift,
        "recommendation": (
            "HALT TRADING: distributions significantly differ"
            if drift else "OK: live performance consistent with backtest"
        ),
    }
    if drift:
        logger.critical(
            "LIVE/BACKTEST DRIFT DETECTED: p=%.4f, "
            "live_mean=%.4f, bt_mean=%.4f",
            p_value, live_rets.mean(), bt_rets.mean(),
        )
    return result
