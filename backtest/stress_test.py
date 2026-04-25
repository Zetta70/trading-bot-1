"""
Stress Test — evaluate the strategy during known crisis windows.

Most strategies look great in bull markets and collapse in stress.
Explicit evaluation of crisis periods is mandatory before real capital.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CrisisWindow:
    name: str
    start: str
    end: str
    description: str


# Canonical KR equity stress windows.
STRESS_WINDOWS_KR = [
    CrisisWindow(
        "covid_crash", "2020-02-15", "2020-04-15",
        "COVID-19 panic: KOSPI -35% in ~5 weeks",
    ),
    CrisisWindow(
        "covid_recovery", "2020-04-15", "2020-12-31",
        "Extreme recovery bull run",
    ),
    CrisisWindow(
        "2022_bear", "2022-01-01", "2022-10-31",
        "Inflation / rate hike bear: KOSPI -30%",
    ),
    CrisisWindow(
        "2024_aug_flash", "2024-07-25", "2024-08-15",
        "Yen carry unwind: 1-day -8% KOSPI",
    ),
]

# Canonical US equity stress windows.
STRESS_WINDOWS_US = [
    CrisisWindow(
        "covid_crash_us", "2020-02-15", "2020-04-15",
        "S&P500 -34% in 5 weeks",
    ),
    CrisisWindow(
        "2022_bear_us", "2022-01-01", "2022-10-31",
        "Tech-led bear: NASDAQ -35%",
    ),
    CrisisWindow(
        "2023_banking", "2023-03-01", "2023-03-31",
        "SVB / Credit Suisse banking panic",
    ),
]


def run_stress_test(
    backtester,  # WalkForwardBacktest instance
    ticker_ohlcv: dict[str, pd.DataFrame],
    index_df: pd.DataFrame | None,
    windows: list[CrisisWindow],
) -> list[dict]:
    """
    For each crisis window, slice the data and run a focused backtest.

    Note: WalkForwardBacktest needs >train_window + purge bars to make
    even a single fold. Many crisis windows are too short, so the
    function records an error and moves on rather than crashing.
    """
    from backtest.metrics import sharpe_ratio, max_drawdown
    from backtest.simulator import TradingSimulator

    results = []
    for win in windows:
        logger.info(
            "Stress test: %s (%s ~ %s)", win.name, win.start, win.end,
        )
        start_ts = pd.Timestamp(win.start)
        end_ts = pd.Timestamp(win.end)

        sliced = {}
        for ticker, ohlcv in ticker_ohlcv.items():
            mask = (ohlcv.index >= start_ts) & (ohlcv.index <= end_ts)
            s = ohlcv[mask]
            if len(s) >= 30:
                sliced[ticker] = s

        if not sliced:
            logger.warning("No data for %s", win.name)
            results.append({"window": win.name, "error": "no data"})
            continue

        index_sliced = None
        if index_df is not None:
            mask = (index_df.index >= start_ts) & (index_df.index <= end_ts)
            index_sliced = index_df[mask]

        try:
            sim = TradingSimulator()
            out = backtester.run(sliced, index_sliced, simulator=sim)
            sim_after = out["simulator"]

            equity_series = pd.Series(
                [e["equity"] for e in sim_after.daily_equity]
            )
            returns = equity_series.pct_change().dropna()

            results.append({
                "window": win.name,
                "description": win.description,
                "start": win.start,
                "end": win.end,
                "sharpe": sharpe_ratio(returns),
                "total_return": (
                    float(equity_series.iloc[-1] / equity_series.iloc[0] - 1)
                    if len(equity_series) > 1 else 0.0
                ),
                "max_dd": max_drawdown(returns)["max_dd"],
                "n_trades": len(sim_after.trades),
            })
        except Exception as e:
            logger.error("Stress test %s failed: %s", win.name, e)
            results.append({"window": win.name, "error": str(e)})

    return results
