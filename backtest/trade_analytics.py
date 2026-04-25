"""
Trade-level Analytics — find patterns that separate winners from losers.

Insights:
  - Are winners and losers similar size? (If losers are larger, risk
    management is failing.)
  - Holding-time distribution.
  - Performance by month / day-of-week.
  - Performance by volatility regime (VKOSPI tercile).
  - Exit-reason profitability (signal vs ATR stop vs time-exit).
"""

from __future__ import annotations

import logging

import numpy as np  # noqa: F401
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_trades(
    trade_log: pd.DataFrame,
    equity_log: pd.DataFrame | None = None,
    vol_index_df: pd.DataFrame | None = None,
) -> dict:
    """
    Comprehensive trade-level diagnostics.

    Parameters
    ----------
    trade_log : DataFrame
        Required columns: entry_date, exit_date, return_pct, holding_days.
        Optional: ticker, exit_reason.
    equity_log : optional portfolio equity time series (unused at present
        but accepted so future per-day overlays can be added cheaply).
    vol_index_df : optional VKOSPI/VIX (close column) for regime split.
    """
    df = trade_log.copy()
    if df.empty:
        return {"error": "no trades"}

    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["win"] = df["return_pct"] > 0

    winners = df[df["win"]]
    losers = df[~df["win"]]

    loss_sum = float(losers["return_pct"].sum())
    profit_factor = (
        float(winners["return_pct"].sum() / abs(loss_sum))
        if len(losers) and loss_sum != 0
        else float("inf")
    )

    report: dict = {
        "n_trades": len(df),
        "win_rate": float(df["win"].mean()),
        "avg_win_pct": (
            float(winners["return_pct"].mean()) if len(winners) else 0.0
        ),
        "avg_loss_pct": (
            float(losers["return_pct"].mean()) if len(losers) else 0.0
        ),
        "profit_factor": profit_factor,
        "expectancy_pct": float(df["return_pct"].mean()),
        "best_trade": float(df["return_pct"].max()),
        "worst_trade": float(df["return_pct"].min()),
        "avg_holding_days": float(df["holding_days"].mean()),
    }

    if "exit_reason" in df.columns:
        report["exit_reasons"] = df.groupby("exit_reason")[
            "return_pct"
        ].agg(["count", "mean", "sum"]).to_dict("index")

    df["dow"] = df["entry_date"].dt.day_name()
    report["by_day_of_week"] = df.groupby("dow")["return_pct"].agg(
        ["count", "mean"],
    ).to_dict("index")

    df["month"] = df["entry_date"].dt.month
    report["by_month"] = df.groupby("month")["return_pct"].agg(
        ["count", "mean"],
    ).to_dict("index")

    bins = [0, 1, 3, 7, 15, 30, float("inf")]
    labels = ["0d", "1-2d", "3-7d", "8-15d", "16-30d", "30d+"]
    df["hold_bucket"] = pd.cut(df["holding_days"], bins=bins, labels=labels)
    report["by_holding_period"] = df.groupby(
        "hold_bucket", observed=False,
    )["return_pct"].agg(["count", "mean", "sum"]).to_dict("index")

    if vol_index_df is not None and not vol_index_df.empty:
        vix = vol_index_df["close"]
        terciles = vix.quantile([1 / 3, 2 / 3]).values

        def _regime(entry_date):
            try:
                level = vix.asof(entry_date)
                if pd.isna(level):
                    return "unknown"
                if level <= terciles[0]:
                    return "low_vol"
                if level <= terciles[1]:
                    return "mid_vol"
                return "high_vol"
            except Exception:
                return "unknown"

        df["regime"] = df["entry_date"].apply(_regime)
        report["by_regime"] = df.groupby("regime")["return_pct"].agg(
            ["count", "mean", "sum"],
        ).to_dict("index")

    warnings = []
    if (
        report["avg_win_pct"] > 0
        and abs(report["avg_loss_pct"]) > report["avg_win_pct"] * 2
    ):
        warnings.append(
            "Losers are >2x size of winners — risk mgmt may be failing"
        )
    if report["win_rate"] > 0.7:
        warnings.append(
            "Win rate >70% is suspicious in finance — check for "
            "lookahead bias"
        )
    if len(df) < 30:
        warnings.append(
            f"Only {len(df)} trades — too few for statistical confidence"
        )
    report["warnings"] = warnings

    return report
