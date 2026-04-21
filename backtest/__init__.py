"""
Backtest Package — walk-forward backtesting for KIS trading strategies.

Modules
-------
data_loader     : Load historical OHLCV from FinanceDataReader / pykrx.
simulator       : Realistic trade execution with costs and position limits.
walk_forward    : Purged walk-forward cross-validation engine.
metrics         : Performance metrics (CAGR, Sharpe, MDD, etc.).
plotter         : Result visualizations (equity curve, heatmap, etc.).
run_backtest    : CLI entry point.
"""
