"""
Purged Walk-Forward Backtesting Engine.

Implements time-series cross-validation with:
  - Fixed-size train/test windows sliding forward
  - Purge gap between train and test to prevent target leakage
  - Per-fold model training with early stopping (raw features; LightGBM is tree-based)
  - Signal generation → Simulator execution on test period

This is the main orchestration module that ties together features.py,
model.py, and simulator.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from model import (
    StockPredictor,
    EnsemblePredictor,
    MetaLabeler,
    make_target,
    make_primary_label,
    make_meta_label,
)
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
        Target label look-ahead period (max holding bars for triple barrier).
    pt_atr_mult : float
        Profit-target barrier width in ATR units.
    sl_atr_mult : float
        Stop-loss barrier width in ATR units.
    atr_period : int
        ATR lookback used by the triple-barrier labeller.
    """

    def __init__(
        self,
        train_window: int = 756,
        test_window: int = 63,
        step: int | None = None,
        purge_days: int = 6,
        horizon: int = 10,
        pt_atr_mult: float = 2.0,
        sl_atr_mult: float = 1.0,
        atr_period: int = 14,
        use_ensemble: bool = True,
        use_meta_labeling: bool = True,
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step or test_window
        self.purge_days = max(purge_days, horizon + 1)
        self.horizon = horizon
        self.pt_atr_mult = pt_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.atr_period = atr_period
        self.use_ensemble = use_ensemble
        self.use_meta_labeling = use_meta_labeling

    def run(
        self,
        ticker_ohlcv: dict[str, pd.DataFrame],
        index_df: pd.DataFrame | None = None,
        simulator: TradingSimulator | None = None,
        vol_index_df: pd.DataFrame | None = None,
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
        ticker_targets_primary: dict[str, pd.Series] = {}
        ticker_targets_meta: dict[str, pd.Series] = {}
        ticker_returns: dict[str, pd.Series] = {}

        for ticker, ohlcv in ticker_ohlcv.items():
            feat_df = add_features(ohlcv, index_df, vol_index_df)
            if self.use_meta_labeling:
                ticker_targets_primary[ticker] = make_primary_label(
                    ohlcv,
                    horizon=self.horizon,
                    pt_atr_mult=self.pt_atr_mult,
                    sl_atr_mult=self.sl_atr_mult,
                )
                ticker_targets_meta[ticker] = make_meta_label(
                    ohlcv,
                    horizon=self.horizon,
                    pt_atr_mult=self.pt_atr_mult,
                    sl_atr_mult=self.sl_atr_mult,
                )
            else:
                ticker_targets[ticker] = make_target(
                    ohlcv,
                    horizon=self.horizon,
                    pt_atr_mult=self.pt_atr_mult,
                    sl_atr_mult=self.sl_atr_mult,
                    atr_period=self.atr_period,
                )
            ticker_features[ticker] = feat_df
            ticker_returns[ticker] = ohlcv["close"].pct_change()

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
        has_vol_index = vol_index_df is not None and not vol_index_df.empty
        feat_cols = feature_columns(
            include_market=has_market,
            include_regime=True,
            include_vol_index=has_vol_index,
        )

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
            X_train_parts: list[pd.DataFrame] = []
            y_train_parts: list[pd.Series] = []
            y_primary_parts: list[pd.Series] = []
            y_meta_parts: list[pd.Series] = []

            for ticker in ticker_ohlcv:
                feat = ticker_features[ticker]
                available = [c for c in feat_cols if c in feat.columns]
                train_feat = feat.loc[feat.index.isin(train_dates), available]

                if self.use_meta_labeling:
                    yp = ticker_targets_primary[ticker]
                    ym = ticker_targets_meta[ticker]
                    yp = yp.loc[yp.index.isin(train_dates)]
                    ym = ym.loc[ym.index.isin(train_dates)]
                    combined = train_feat.join(
                        yp.rename("y_primary"),
                    ).join(ym.rename("y_meta")).dropna()
                    if len(combined) > 0:
                        X_train_parts.append(combined[available])
                        y_primary_parts.append(combined["y_primary"])
                        y_meta_parts.append(combined["y_meta"])
                else:
                    tgt = ticker_targets[ticker].loc[
                        ticker_targets[ticker].index.isin(train_dates)
                    ]
                    combined = train_feat.join(tgt).dropna()
                    if len(combined) > 0:
                        X_train_parts.append(combined[available])
                        y_train_parts.append(combined["target"])

            if not X_train_parts:
                logger.warning("Fold %d: no training data. Skipping.", fold_idx)
                test_start_pos += self.step
                continue

            X_train_all = pd.concat(X_train_parts)
            available = list(X_train_all.columns)

            # Train/val split: last 20% for early stopping.
            val_size = max(1, int(len(X_train_all) * 0.2))
            X_tr = X_train_all.iloc[:-val_size]
            X_va = X_train_all.iloc[-val_size:]

            if self.use_meta_labeling:
                y_primary_all = pd.concat(y_primary_parts)
                y_meta_all = pd.concat(y_meta_parts)
                yp_tr = y_primary_all.iloc[:-val_size]
                yp_va = y_primary_all.iloc[-val_size:]
                ym_tr = y_meta_all.iloc[:-val_size]
                ym_va = y_meta_all.iloc[-val_size:]

                model: "StockPredictor | EnsemblePredictor | MetaLabeler" = (
                    MetaLabeler()
                )
                model.train(
                    X_tr, yp_tr, ym_tr,
                    X_va, yp_va, ym_va,
                )
            else:
                y_train_all = pd.concat(y_train_parts)
                y_tr = y_train_all.iloc[:-val_size]
                y_va = y_train_all.iloc[-val_size:]

                if self.use_ensemble:
                    model = EnsemblePredictor(task="classification")
                    model.train(X_tr, y_tr, X_va, y_va)
                else:
                    model = StockPredictor(task="classification")
                    model.train(
                        X_tr, y_tr, X_va, y_va, early_stopping_rounds=50,
                    )

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
                    # Pull ATR from the feature DataFrame (atr_norm is
                    # ATR/close, so ATR = atr_norm * close).
                    feat = ticker_features[ticker]
                    atr_val = 0.0
                    if date in feat.index and "atr_norm" in feat.columns:
                        atr_norm = feat.loc[date, "atr_norm"]
                        if pd.notna(atr_norm):
                            atr_val = float(atr_norm) * float(row["close"])

                    ticker_data_today[ticker] = {
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "atr": atr_val,
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
                        feat_row = feat.loc[[prev_date], available]
                        if feat_row.isna().any(axis=1).iloc[0]:
                            signal_probs[ticker] = None
                        else:
                            prob = model.predict_proba(feat_row)[0]
                            signal_probs[ticker] = prob
                    else:
                        signal_probs[ticker] = None

                if ticker_data_today:
                    returns_as_of_date = {
                        t: ret.loc[ret.index < date]
                        for t, ret in ticker_returns.items()
                    }
                    simulator.step_portfolio(
                        date, ticker_data_today, signal_probs, close_prices,
                        ticker_returns=returns_as_of_date,
                    )

            fold_trades = len(simulator.trades) - fold_trades_before
            fold_results.append({
                "fold": fold_idx,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "n_trades": fold_trades,
                "best_iteration": getattr(
                    getattr(model, "model", None), "best_iteration", 0,
                ),
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
