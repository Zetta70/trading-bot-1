"""
Walk-Forward Analysis — detect regime-specific failures.

A strategy with overall Sharpe 2.0 but Sharpe -0.5 in 2/10 folds is
likely overfitted. Consistent fold performance is the real test.
"""

from __future__ import annotations

import logging

import numpy as np  # noqa: F401  (re-exported through future analyses)
import pandas as pd

logger = logging.getLogger(__name__)


def fold_consistency_report(
    fold_results: list[dict],
    equity_log: pd.DataFrame,
) -> dict:
    """
    Per-fold performance report.

    Inputs
    ------
    fold_results : list of dicts from WalkForwardBacktest
        Each: fold, test_start, test_end, n_trades, best_iteration
    equity_log : DataFrame with 'timestamp' and 'equity'
        Optional 'ticker' column (uses PORTFOLIO rows when present).

    Returns
    -------
    dict with per-fold Sharpe / return / DD plus a stability score
    (fraction of folds with positive Sharpe).
    """
    from backtest.metrics import sharpe_ratio, max_drawdown

    if equity_log is None or equity_log.empty or not fold_results:
        return {"error": "insufficient data"}

    equity_log = equity_log.copy()
    equity_log["timestamp"] = pd.to_datetime(equity_log["timestamp"])
    equity_log = equity_log.set_index("timestamp").sort_index()

    if "ticker" in equity_log.columns:
        portfolio = equity_log[equity_log["ticker"] == "PORTFOLIO"]
        equity_series = (
            portfolio["equity"] if not portfolio.empty
            else equity_log["equity"]
        )
    else:
        equity_series = equity_log["equity"]

    returns = equity_series.pct_change().dropna()

    fold_stats = []
    for fr in fold_results:
        test_start = pd.Timestamp(fr["test_start"])
        test_end = pd.Timestamp(fr["test_end"])
        mask = (returns.index >= test_start) & (returns.index <= test_end)
        fold_rets = returns[mask]
        if len(fold_rets) < 5:
            continue

        fold_stats.append({
            "fold": fr["fold"],
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "sharpe": sharpe_ratio(fold_rets),
            "total_return": float((1 + fold_rets).prod() - 1),
            "max_dd": max_drawdown(fold_rets)["max_dd"],
            "n_trades": fr.get("n_trades", 0),
        })

    df = pd.DataFrame(fold_stats)
    if df.empty:
        return {"error": "no fold overlap with equity log"}

    n_total = len(df)
    n_positive = int((df["sharpe"] > 0).sum())
    stability = n_positive / n_total
    profitable_folds_pct = int((df["total_return"] > 0).sum()) / n_total

    return {
        "n_folds": n_total,
        "n_positive_sharpe": n_positive,
        "stability_score": stability,
        "profitable_folds_pct": profitable_folds_pct,
        "sharpe_mean": float(df["sharpe"].mean()),
        "sharpe_std": float(df["sharpe"].std()),
        "sharpe_min": float(df["sharpe"].min()),
        "sharpe_max": float(df["sharpe"].max()),
        "worst_fold": df.loc[df["sharpe"].idxmin()].to_dict(),
        "best_fold": df.loc[df["sharpe"].idxmax()].to_dict(),
        "all_folds": df.to_dict("records"),
    }
