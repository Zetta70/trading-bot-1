"""
Purged Walk-Forward Backtesting Engine.

Implements time-series cross-validation with:
  - Fixed-size train/test windows sliding forward
  - Purge gap between train and test to prevent target leakage
  - Per-fold StandardScaler fit (train only)
  - Per-fold model training with early stopping
  - Signal generation → Simulator execution on test period

This is the main orchestration module that ties together features.py,
model.py, and simulator.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from model import StockPredictor, make_target
from features import add_features, feature_columns
from backtest.simulator import TradingSimulator

logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """
    Purged walk-forward cross-validation backtester.

    Parameters
    ----------
    train_window : int
        Number of trading days for each training fold. Default 756 (~3 years).
    test_window : int
        Number of trading days for each test fold. Default 63 (~3 months).
    step : int
        How far to slide the window each fold. Default = test_window.
    purge_days : int
        Gap between train end and test start to prevent label leakage.
        Must be ≥ horizon + 1.
    horizon : int
        Target label look-ahead period (days).
    target_threshold : float
        Return threshold for binary target labelling.
    """

    def __init__(
        self,
        train_window: int = 756,
        test_window: int = 63,
        step: int | None = None,
        purge_days: int = 6,
        horizon: int = 5,
        target_threshold: float = 0.02,
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step or test_window
        self.purge_days = max(purge_days, horizon + 1)
        self.horizon = horizon
        self.target_threshold = target_threshold

    def run(
        self,
        ticker_ohlcv: dict[str, pd.DataFrame],
        index_df: pd.DataFrame | None = None,
        simulator: TradingSimulator | None = None,
    ) -> dict:
        """
        Execute the full walk-forward backtest.

        Parameters
        ----------
        ticker_ohlcv : dict[str, pd.DataFrame]
            {ticker: OHLCV DataFrame} for each stock to backtest.
        index_df : pd.DataFrame, optional
            KOSPI index OHLCV for market-context features.
        simulator : TradingSimulator, optional
            Pre-configured simulator. If None, uses default settings.

        Returns
        -------
        dict with keys:
            'simulator' : TradingSimulator with all trades and equity
            'fold_results' : list of per-fold summary dicts
            'feature_importance' : averaged feature importance across folds
        """
        if simulator is None:
            simulator = TradingSimulator()
        simulator.reset()

        # ── Prepare features and targets for each ticker ─────────
        ticker_features: dict[str, pd.DataFrame] = {}
        ticker_targets: dict[str, pd.Series] = {}

        for ticker, ohlcv in ticker_ohlcv.items():
            feat_df = add_features(ohlcv, index_df)
            target = make_target(
                ohlcv, horizon=self.horizon, threshold=self.target_threshold,
            )
            ticker_features[ticker] = feat_df
            ticker_targets[ticker] = target

        # ── Determine fold boundaries using first ticker's index ─
        first_ticker = list(ticker_ohlcv.keys())[0]
        all_dates = ticker_features[first_ticker].index
        n_total = len(all_dates)

        min_start = self.train_window + self.purge_days
        if min_start >= n_total:
            raise ValueError(
                f"Not enough data: {n_total} rows, need at least "
                f"{min_start} (train={self.train_window} + "
                f"purge={self.purge_days})"
            )

        fold_results = []
        importance_accum: list[pd.DataFrame] = []
        has_market = index_df is not None
        feat_cols = feature_columns(include_market=has_market)

        fold_idx = 0
        test_start_pos = min_start

        while test_start_pos + self.test_window <= n_total:
            fold_idx += 1
            train_end_pos = test_start_pos - self.purge_days
            train_start_pos = max(0, train_end_pos - self.train_window)

            test_end_pos = min(test_start_pos + self.test_window, n_total)

            train_dates = all_dates[train_start_pos:train_end_pos]
            test_dates = all_dates[test_start_pos:test_end_pos]

            logger.info(
                "Fold %d: train %s~%s (%d), purge %d days, test %s~%s (%d)",
                fold_idx,
                train_dates[0].strftime("%Y-%m-%d"),
                train_dates[-1].strftime("%Y-%m-%d"),
                len(train_dates),
                self.purge_days,
                test_dates[0].strftime("%Y-%m-%d"),
                test_dates[-1].strftime("%Y-%m-%d"),
                len(test_dates),
            )

            # ── Collect train/val data across all tickers ────────
            X_train_parts, y_train_parts = [], []

            for ticker in ticker_ohlcv:
                feat = ticker_features[ticker]
                tgt = ticker_targets[ticker]

                # Select available feature columns
                available = [c for c in feat_cols if c in feat.columns]
                train_feat = feat.loc[feat.index.isin(train_dates), available]
                train_tgt = tgt.loc[tgt.index.isin(train_dates)]

                # Align and drop NaN
                combined = train_feat.join(train_tgt).dropna()
                if len(combined) > 0:
                    X_train_parts.append(combined[available])
                    y_train_parts.append(combined["target"])

            if not X_train_parts:
                logger.warning("Fold %d: no training data. Skipping.", fold_idx)
                test_start_pos += self.step
                continue

            X_train_all = pd.concat(X_train_parts)
            y_train_all = pd.concat(y_train_parts)

            # ── Scale features (fit on train only) ───────────────
            scaler = StandardScaler()
            available = list(X_train_all.columns)
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train_all),
                columns=available,
                index=X_train_all.index,
            )

            # ── Split train into train/val (last 20% for early stopping)
            val_size = max(1, int(len(X_train_scaled) * 0.2))
            X_tr = X_train_scaled.iloc[:-val_size]
            y_tr = y_train_all.iloc[:-val_size]
            X_va = X_train_scaled.iloc[-val_size:]
            y_va = y_train_all.iloc[-val_size:]

            # ── Train model ──────────────────────────────────────
            model = StockPredictor(task="classification")
            model.train(X_tr, y_tr, X_va, y_va, early_stopping_rounds=50)

            imp = model.feature_importance()
            imp["fold"] = fold_idx
            importance_accum.append(imp)

            # ── Generate predictions on test period ──────────────
            fold_trades_before = len(simulator.trades)

            for i, date in enumerate(test_dates):
                ticker_data_today: dict[str, dict] = {}
                signal_probs: dict[str, float | None] = {}
                close_prices: dict[str, float] = {}

                for ticker in ticker_ohlcv:
                    ohlcv = ticker_ohlcv[ticker]
                    if date not in ohlcv.index:
                        continue

                    row = ohlcv.loc[date]
                    ticker_data_today[ticker] = {
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                    }
                    close_prices[ticker] = float(row["close"])

                    # Signal = prediction from previous day's features
                    # (first day of test has no signal)
                    if i == 0:
                        signal_probs[ticker] = None
                        continue

                    prev_date = test_dates[i - 1]
                    feat = ticker_features[ticker]
                    if prev_date in feat.index:
                        feat_row = feat.loc[[prev_date], available].fillna(0)
                        feat_scaled = pd.DataFrame(
                            scaler.transform(feat_row),
                            columns=available,
                            index=feat_row.index,
                        )
                        prob = model.predict_proba(feat_scaled)[0]
                        signal_probs[ticker] = prob
                    else:
                        signal_probs[ticker] = None

                if ticker_data_today:
                    simulator.step_portfolio(
                        date, ticker_data_today, signal_probs, close_prices,
                    )

            fold_trades = len(simulator.trades) - fold_trades_before
            fold_results.append({
                "fold": fold_idx,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "n_trades": fold_trades,
                "best_iteration": model.model.best_iteration if model.model else 0,
            })

            test_start_pos += self.step

        # ── Close any remaining open positions at last close ─────
        if ticker_ohlcv:
            last_date = all_dates[-1]
            for ticker in list(simulator.positions.keys()):
                ohlcv = ticker_ohlcv.get(ticker)
                if ohlcv is not None and last_date in ohlcv.index:
                    close_p = float(ohlcv.loc[last_date, "close"])
                    simulator._execute_sell(
                        ticker, last_date, close_p, reason="backtest_end",
                    )

        # ── Average feature importance across folds ──────────────
        if importance_accum:
            all_imp = pd.concat(importance_accum)
            avg_imp = (
                all_imp.groupby("feature")["importance"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
        else:
            avg_imp = pd.DataFrame(columns=["feature", "importance"])

        logger.info(
            "Walk-forward complete: %d folds, %d total trades",
            len(fold_results), len(simulator.trades),
        )

        return {
            "simulator": simulator,
            "fold_results": fold_results,
            "feature_importance": avg_imp,
        }
