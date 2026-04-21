"""
Backtest Runner — CLI entry point.

Usage
-----
    python -m backtest.run_backtest \\
        --start 2019-01-01 --end 2024-12-31 \\
        --tickers 005930,000660,035420 \\
        --horizon 5 --threshold 0.02

All parameters default to .env values (via config.py).
CLI arguments override .env when explicitly provided.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when run as module
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from backtest.data_loader import load_ohlcv, load_kospi
from backtest.simulator import TradingSimulator
from backtest.walk_forward import WalkForwardBacktest
from backtest.metrics import compute_metrics, format_metrics
from backtest.plotter import plot_all

logger = logging.getLogger(__name__)


def parse_args(cfg: Config) -> argparse.Namespace:
    """
    Parse CLI arguments with .env (Config) values as defaults.

    If a CLI argument is not explicitly provided, the .env value is used.
    """
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtest on Korean stocks.",
    )
    parser.add_argument(
        "--start", type=str, default="2019-01-01",
        help="Backtest start date (YYYY-MM-DD). Default: 2019-01-01",
    )
    parser.add_argument(
        "--end", type=str, default="2024-12-31",
        help="Backtest end date (YYYY-MM-DD). Default: 2024-12-31",
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated ticker codes. Default: from .env TICKERS",
    )
    parser.add_argument(
        "--horizon", type=int, default=cfg.bt_horizon,
        help=f"Target horizon in trading days. Default from .env: {cfg.bt_horizon}",
    )
    parser.add_argument(
        "--threshold", type=float, default=cfg.bt_target_threshold,
        help=f"Return threshold for target label. Default from .env: {cfg.bt_target_threshold}",
    )
    parser.add_argument(
        "--train-window", type=int, default=cfg.bt_train_window,
        help=f"Training window in days. Default from .env: {cfg.bt_train_window}",
    )
    parser.add_argument(
        "--test-window", type=int, default=cfg.bt_test_window,
        help=f"Test window in days. Default from .env: {cfg.bt_test_window}",
    )
    parser.add_argument(
        "--initial-capital", type=int, default=cfg.initial_cash,
        help=f"Initial capital in KRW. Default from .env: {cfg.initial_cash:,}",
    )
    parser.add_argument(
        "--threshold-buy", type=float, default=cfg.ml_threshold_buy,
        help=f"ML buy signal threshold. Default from .env: {cfg.ml_threshold_buy}",
    )
    parser.add_argument(
        "--threshold-sell", type=float, default=cfg.ml_threshold_sell,
        help=f"ML sell signal threshold. Default from .env: {cfg.ml_threshold_sell}",
    )
    parser.add_argument(
        "--stop-loss-pct", type=float, default=cfg.stop_loss_pct,
        help=f"Stop-loss %% (negative). Default from .env: {cfg.stop_loss_pct}",
    )
    parser.add_argument(
        "--save-model", type=str, default=None,
        help="Path to save the final trained model (e.g. models/lgbm_v1.pkl)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="backtest_results",
        help="Directory for output files. Default: backtest_results",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip chart generation.",
    )
    return parser.parse_args()


def _log_parameters(args: argparse.Namespace, tickers: list[str]) -> None:
    """Log all active backtest parameters for reproducibility."""
    logger.info("=" * 60)
    logger.info("  BACKTEST PARAMETERS (from .env + CLI overrides)")
    logger.info("=" * 60)
    logger.info("  Period:            %s ~ %s", args.start, args.end)
    logger.info("  Tickers:           %s", ", ".join(tickers))
    logger.info("  Initial Capital:   %s KRW", f"{args.initial_capital:,}")
    logger.info("  ── Walk-Forward ──")
    logger.info("  Train Window:      %d days", args.train_window)
    logger.info("  Test Window:       %d days", args.test_window)
    logger.info("  Horizon:           %d days", args.horizon)
    logger.info("  Target Threshold:  %.4f (%.2f%%)", args.threshold, args.threshold * 100)
    logger.info("  ── Simulator ──")
    logger.info("  ML Threshold BUY:  %.4f", args.threshold_buy)
    logger.info("  ML Threshold SELL: %.4f", args.threshold_sell)
    logger.info("  Stop-Loss:         %.2f%%", args.stop_loss_pct * 100)
    logger.info("=" * 60)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Load .env via Config (single source of truth) ────────────
    cfg = Config()
    args = parse_args(cfg)

    # ── Resolve tickers ──────────────────────────────────────────
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = cfg.tickers

    _log_parameters(args, tickers)

    # ── Load data ────────────────────────────────────────────────
    ticker_ohlcv: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = load_ohlcv(ticker, args.start, args.end)
            if len(df) > 0:
                ticker_ohlcv[ticker] = df
        except Exception as e:
            logger.error("Failed to load %s: %s", ticker, e)

    if not ticker_ohlcv:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)

    try:
        index_df = load_kospi(args.start, args.end)
    except Exception as e:
        logger.warning("Failed to load KOSPI index: %s. Continuing without.", e)
        index_df = None

    # ── Configure simulator with .env values ─────────────────────
    simulator = TradingSimulator(
        initial_capital=args.initial_capital,
        max_positions=min(len(tickers), 10),
        threshold_buy=args.threshold_buy,
        threshold_sell=args.threshold_sell,
        stop_loss_pct=args.stop_loss_pct,
    )

    # ── Run walk-forward backtest ────────────────────────────────
    engine = WalkForwardBacktest(
        train_window=args.train_window,
        test_window=args.test_window,
        horizon=args.horizon,
        target_threshold=args.threshold,
        purge_days=args.horizon + 1,
    )

    results = engine.run(ticker_ohlcv, index_df, simulator)
    sim: TradingSimulator = results["simulator"]

    # ── Output results ───────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV logs (same format as live bot)
    equity_log = sim.get_equity_log()
    trade_log = sim.get_trade_log()

    equity_log.to_csv(output_dir / "equity_log.csv", index=False)
    trade_log.to_csv(output_dir / "trade_log.csv", index=False)

    # Compute and print metrics
    benchmark = index_df if index_df is not None else None
    metrics = compute_metrics(equity_log, trade_log, benchmark)
    report = format_metrics(metrics)
    print(report)

    with open(output_dir / "metrics.txt", "w") as f:
        f.write(report)

    # Fold summary
    fold_df = pd.DataFrame(results["fold_results"])
    if not fold_df.empty:
        fold_df.to_csv(output_dir / "fold_summary.csv", index=False)
        print(f"\nFold summary ({len(fold_df)} folds):")
        print(fold_df.to_string(index=False))

    # Feature importance
    fi = results["feature_importance"]
    if not fi.empty:
        fi.to_csv(output_dir / "feature_importance.csv", index=False)
        print("\nTop 10 features:")
        print(fi.head(10).to_string(index=False))

    # ── Charts ───────────────────────────────────────────────────
    if not args.no_plot:
        try:
            plot_all(equity_log, trade_log, fi, benchmark, str(output_dir))
        except Exception as e:
            logger.error("Plotting failed: %s", e)

    # ── Save model (optional) ────────────────────────────────────
    if args.save_model:
        # Re-train on ALL data for deployment
        logger.info("Re-training on full dataset for deployment...")
        from features import add_features, feature_columns
        from model import StockPredictor, make_target
        from sklearn.preprocessing import StandardScaler

        all_X, all_y = [], []
        has_market = index_df is not None
        feat_cols = feature_columns(include_market=has_market)

        for ticker, ohlcv in ticker_ohlcv.items():
            feat = add_features(ohlcv, index_df)
            tgt = make_target(ohlcv, horizon=args.horizon, threshold=args.threshold)
            available = [c for c in feat_cols if c in feat.columns]
            combined = feat[available].join(tgt).dropna()
            if len(combined) > 0:
                all_X.append(combined[available])
                all_y.append(combined["target"])

        if all_X:
            X_full = pd.concat(all_X)
            y_full = pd.concat(all_y)

            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_full),
                columns=X_full.columns,
                index=X_full.index,
            )

            val_size = max(1, int(len(X_scaled) * 0.1))
            model = StockPredictor(task="classification")
            model.train(
                X_scaled.iloc[:-val_size], y_full.iloc[:-val_size],
                X_scaled.iloc[-val_size:], y_full.iloc[-val_size:],
            )
            model.save(args.save_model)
            logger.info("Model saved to %s", args.save_model)

    logger.info("Backtest complete. Results saved to %s/", args.output_dir)


if __name__ == "__main__":
    main()
