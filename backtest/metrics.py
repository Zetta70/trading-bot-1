"""
Performance Metrics Calculator.

Computes standard quant metrics from backtest results:
  - CAGR, Sharpe, Sortino
  - MDD and MDD duration
  - Win rate, profit factor, expectancy
  - Benchmark-relative metrics (excess return, information ratio)
  - Monthly returns table
  - Multi-market integration (KR + US with FX adjustment)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    equity_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    risk_free_rate: float = 0.03,
) -> dict:
    """
    Compute all performance metrics.

    Parameters
    ----------
    equity_df : pd.DataFrame
        Daily equity log with columns: timestamp, equity.
        Should contain one row per day (PORTFOLIO-level).
    trade_df : pd.DataFrame
        Completed trade log from simulator.
    benchmark_df : pd.DataFrame, optional
        Benchmark (KOSPI) daily close for relative metrics.
    risk_free_rate : float
        Annual risk-free rate for Sharpe/Sortino. Default 3%.

    Returns
    -------
    dict : All computed metrics.
    """
    metrics: dict = {}

    # ── Prepare daily returns series ─────────────────────────────
    equity = equity_df.set_index("timestamp")["equity"].sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]

    if len(equity) < 2:
        return {"error": "Insufficient equity data"}

    daily_ret = equity.pct_change().dropna()
    total_days = (equity.index[-1] - equity.index[0]).days
    years = max(total_days / 365.25, 1 / 365.25)

    # ── Return metrics ───────────────────────────────────────────
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    metrics["total_return"] = total_return
    metrics["cagr"] = (1 + total_return) ** (1 / years) - 1

    # ── Risk metrics ─────────────────────────────────────────────
    ann_factor = np.sqrt(252)
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    excess_ret = daily_ret - daily_rf
    std = daily_ret.std()
    metrics["annual_volatility"] = std * ann_factor
    metrics["sharpe"] = (
        excess_ret.mean() / excess_ret.std() * ann_factor
        if excess_ret.std() > 0 else 0.0
    )

    downside = excess_ret[excess_ret < 0]
    downside_std = downside.std() if len(downside) > 0 else 0.0
    metrics["sortino"] = (
        excess_ret.mean() / downside_std * ann_factor
        if downside_std > 0 else 0.0
    )

    # ── Maximum Drawdown ─────────────────────────────────────────
    cum_max = equity.cummax()
    drawdown = (equity - cum_max) / cum_max
    metrics["mdd"] = drawdown.min()

    # MDD duration (days)
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        dd_groups = (~in_drawdown).cumsum()
        dd_lengths = in_drawdown.groupby(dd_groups).sum()
        metrics["mdd_duration_days"] = int(dd_lengths.max())
    else:
        metrics["mdd_duration_days"] = 0

    # ── Trade metrics ────────────────────────────────────────────
    if len(trade_df) > 0 and "return_pct" in trade_df.columns:
        returns = trade_df["return_pct"]
        winners = returns[returns > 0]
        losers = returns[returns <= 0]

        metrics["n_trades"] = len(returns)
        metrics["win_rate"] = len(winners) / len(returns) if len(returns) > 0 else 0
        metrics["avg_win"] = winners.mean() if len(winners) > 0 else 0
        metrics["avg_loss"] = losers.mean() if len(losers) > 0 else 0

        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
        metrics["profit_factor"] = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        wr = metrics["win_rate"]
        metrics["expectancy"] = (
            metrics["avg_win"] * wr + metrics["avg_loss"] * (1 - wr)
        )

        if "holding_days" in trade_df.columns:
            metrics["avg_holding_days"] = trade_df["holding_days"].mean()

        if "exit_reason" in trade_df.columns:
            sl_count = (trade_df["exit_reason"] == "stop_loss").sum()
            metrics["stop_loss_count"] = int(sl_count)
            metrics["stop_loss_pct"] = sl_count / len(trade_df)
    else:
        metrics["n_trades"] = 0

    # ── Benchmark-relative metrics ───────────────────────────────
    if benchmark_df is not None:
        bm_close = benchmark_df["close"].sort_index()
        bm_close = bm_close.reindex(equity.index, method="ffill")
        bm_ret = bm_close.pct_change().dropna()

        common_idx = daily_ret.index.intersection(bm_ret.index)
        if len(common_idx) > 1:
            strat = daily_ret.loc[common_idx]
            bench = bm_ret.loc[common_idx]

            bm_total = bm_close.iloc[-1] / bm_close.iloc[0] - 1
            metrics["benchmark_return"] = bm_total
            metrics["excess_return"] = total_return - bm_total

            tracking_error = (strat - bench).std() * ann_factor
            metrics["tracking_error"] = tracking_error
            metrics["information_ratio"] = (
                (strat - bench).mean() * 252 / tracking_error
                if tracking_error > 0 else 0.0
            )

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Multi-Market Integration (KR + US)
# ═══════════════════════════════════════════════════════════════════════

def compute_integrated_metrics(
    kr_equity_df: pd.DataFrame | None,
    kr_trade_df: pd.DataFrame | None,
    us_equity_df: pd.DataFrame | None,
    us_trade_df: pd.DataFrame | None,
    fx_df: pd.DataFrame | None = None,
    default_usdkrw: float = 1380.0,
    risk_free_rate: float = 0.03,
) -> dict:
    """
    Compute unified metrics across KR and US markets.

    US equity is converted to KRW using the daily exchange rate.
    FX impact (gain/loss from currency movement) is calculated separately.

    Parameters
    ----------
    kr_equity_df : Daily equity in KRW.
    kr_trade_df : KR trade log.
    us_equity_df : Daily equity in USD.
    us_trade_df : US trade log.
    fx_df : Daily USD/KRW rate (column: 'close'). Optional.
    default_usdkrw : Fallback rate when fx_df is unavailable.
    risk_free_rate : Annual risk-free rate.

    Returns
    -------
    dict with keys:
        'integrated' : combined metrics
        'kr' : KR-only metrics (if available)
        'us' : US-only metrics in USD (if available)
        'fx_impact' : FX gain/loss analysis
    """
    result: dict = {}

    # ── KR metrics (native KRW) ──────────────────────────────────
    if kr_equity_df is not None and len(kr_equity_df) > 0:
        result["kr"] = compute_metrics(
            kr_equity_df,
            kr_trade_df if kr_trade_df is not None else pd.DataFrame(),
            risk_free_rate=risk_free_rate,
        )
    else:
        result["kr"] = {}

    # ── US metrics (native USD) ──────────────────────────────────
    if us_equity_df is not None and len(us_equity_df) > 0:
        result["us"] = compute_metrics(
            us_equity_df,
            us_trade_df if us_trade_df is not None else pd.DataFrame(),
            risk_free_rate=risk_free_rate,
        )
    else:
        result["us"] = {}

    # ── Convert US equity to KRW and merge ───────────────────────
    if us_equity_df is not None and len(us_equity_df) > 0:
        us_equity = us_equity_df.copy()
        us_equity["timestamp"] = pd.to_datetime(us_equity["timestamp"])

        if fx_df is not None and len(fx_df) > 0:
            fx_rate = fx_df["close"].sort_index()
            # Align FX rate to equity dates
            us_equity = us_equity.set_index("timestamp")
            fx_aligned = fx_rate.reindex(us_equity.index, method="ffill").fillna(
                default_usdkrw,
            )
            us_equity["equity_krw"] = us_equity["equity"] * fx_aligned
            us_equity = us_equity.reset_index()

            # FX impact calculation
            fx_start = fx_aligned.iloc[0]
            fx_end = fx_aligned.iloc[-1]
            fx_change_pct = (fx_end - fx_start) / fx_start
            us_equity_start_usd = us_equity_df.set_index("timestamp")["equity"].iloc[0]
            us_equity_end_usd = us_equity_df.set_index("timestamp")["equity"].iloc[-1]

            # FX impact = what US portfolio would be worth at end rate vs start rate
            fx_gain_krw = us_equity_end_usd * (fx_end - fx_start)

            result["fx_impact"] = {
                "usdkrw_start": float(fx_start),
                "usdkrw_end": float(fx_end),
                "usdkrw_change_pct": float(fx_change_pct),
                "fx_gain_krw": float(fx_gain_krw),
            }
        else:
            us_equity = us_equity.copy()
            us_equity["equity_krw"] = us_equity["equity"] * default_usdkrw
            result["fx_impact"] = {
                "usdkrw_start": default_usdkrw,
                "usdkrw_end": default_usdkrw,
                "usdkrw_change_pct": 0.0,
                "fx_gain_krw": 0.0,
            }

        us_equity_krw = us_equity[["timestamp", "equity_krw"]].rename(
            columns={"equity_krw": "equity"},
        )
    else:
        us_equity_krw = None
        result["fx_impact"] = {}

    # ── Merge KR + US (both in KRW) ──────────────────────────────
    dfs_to_merge = []
    if kr_equity_df is not None and len(kr_equity_df) > 0:
        kr = kr_equity_df[["timestamp", "equity"]].copy()
        kr["timestamp"] = pd.to_datetime(kr["timestamp"])
        dfs_to_merge.append(kr.set_index("timestamp")["equity"])

    if us_equity_krw is not None and len(us_equity_krw) > 0:
        us_krw = us_equity_krw.copy()
        us_krw["timestamp"] = pd.to_datetime(us_krw["timestamp"])
        dfs_to_merge.append(us_krw.set_index("timestamp")["equity"])

    if dfs_to_merge:
        # Sum daily equity across markets, forward-fill gaps
        combined = pd.concat(dfs_to_merge, axis=1)
        combined.columns = range(len(combined.columns))
        combined = combined.ffill().fillna(0)
        combined["total"] = combined.sum(axis=1)

        merged_equity = pd.DataFrame({
            "timestamp": combined.index,
            "equity": combined["total"].values,
        })

        # Merge trade logs
        trade_parts = []
        if kr_trade_df is not None and len(kr_trade_df) > 0:
            t = kr_trade_df.copy()
            t["market"] = "KR"
            trade_parts.append(t)
        if us_trade_df is not None and len(us_trade_df) > 0:
            t = us_trade_df.copy()
            t["market"] = "US"
            trade_parts.append(t)
        merged_trades = pd.concat(trade_parts) if trade_parts else pd.DataFrame()

        result["integrated"] = compute_metrics(
            merged_equity, merged_trades, risk_free_rate=risk_free_rate,
        )
    else:
        result["integrated"] = {}

    return result


# ═══════════════════════════════════════════════════════════════════════
# Monthly Returns
# ═══════════════════════════════════════════════════════════════════════

def monthly_returns(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly return table.

    Returns
    -------
    pd.DataFrame
        Index: year, Columns: month (1-12), Values: monthly return (%).
    """
    equity = equity_df.set_index("timestamp")["equity"].sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]

    monthly = equity.resample("ME").last()
    monthly_ret = monthly.pct_change().dropna()

    table = pd.DataFrame(
        index=sorted(monthly_ret.index.year.unique()),
        columns=range(1, 13),
        dtype=float,
    )

    for date, ret in monthly_ret.items():
        table.loc[date.year, date.month] = ret * 100

    table.index.name = "Year"
    return table


# ═══════════════════════════════════════════════════════════════════════
# Formatters
# ═══════════════════════════════════════════════════════════════════════

def format_metrics(metrics: dict) -> str:
    """Format metrics dict as a readable string report."""
    lines = [
        "═" * 55,
        "  BACKTEST PERFORMANCE REPORT",
        "═" * 55,
        "",
        "── Returns ──────────────────────────────────────",
        f"  Total Return:      {metrics.get('total_return', 0):.2%}",
        f"  CAGR:              {metrics.get('cagr', 0):.2%}",
        f"  Annual Volatility: {metrics.get('annual_volatility', 0):.2%}",
        "",
        "── Risk-adjusted ───────────────────────────────",
        f"  Sharpe Ratio:      {metrics.get('sharpe', 0):.3f}",
        f"  Sortino Ratio:     {metrics.get('sortino', 0):.3f}",
        f"  Max Drawdown:      {metrics.get('mdd', 0):.2%}",
        f"  MDD Duration:      {metrics.get('mdd_duration_days', 0)} days",
        "",
        "── Trades ──────────────────────────────────────",
        f"  Total Trades:      {metrics.get('n_trades', 0)}",
        f"  Win Rate:          {metrics.get('win_rate', 0):.1%}",
        f"  Avg Win:           {metrics.get('avg_win', 0):.2%}",
        f"  Avg Loss:          {metrics.get('avg_loss', 0):.2%}",
        f"  Profit Factor:     {metrics.get('profit_factor', 0):.2f}",
        f"  Expectancy:        {metrics.get('expectancy', 0):.4f}",
        f"  Avg Holding:       {metrics.get('avg_holding_days', 0):.1f} days",
        f"  Stop-Loss Exits:   {metrics.get('stop_loss_count', 0)}"
        f" ({metrics.get('stop_loss_pct', 0):.1%})",
    ]

    if "benchmark_return" in metrics:
        lines += [
            "",
            "── Benchmark (KOSPI) ───────────────────────────",
            f"  Benchmark Return:  {metrics['benchmark_return']:.2%}",
            f"  Excess Return:     {metrics['excess_return']:.2%}",
            f"  Information Ratio: {metrics.get('information_ratio', 0):.3f}",
        ]

    lines.append("")
    lines.append("═" * 55)
    return "\n".join(lines)


def format_integrated_metrics(result: dict) -> str:
    """
    Format multi-market metrics as a readable report.

    Parameters
    ----------
    result : Output of compute_integrated_metrics().
    """
    lines = [
        "═" * 60,
        "  INTEGRATED BACKTEST REPORT (KR + US)",
        "═" * 60,
    ]

    # ── Integrated ───────────────────────────────────────────────
    intg = result.get("integrated", {})
    if intg and "error" not in intg:
        lines += [
            "",
            "── Combined (KRW) ────────────────────────────────",
            f"  Total Return:   {intg.get('total_return', 0):.2%}",
            f"  CAGR:           {intg.get('cagr', 0):.2%}",
            f"  Sharpe:         {intg.get('sharpe', 0):.3f}",
            f"  MDD:            {intg.get('mdd', 0):.2%}",
            f"  Total Trades:   {intg.get('n_trades', 0)}",
        ]

    # ── KR ────────────────────────────────────────────────────────
    kr = result.get("kr", {})
    if kr and "error" not in kr:
        lines += [
            "",
            "── KR Market (KRW) ───────────────────────────────",
            f"  Total Return:   {kr.get('total_return', 0):.2%}",
            f"  Sharpe:         {kr.get('sharpe', 0):.3f}",
            f"  MDD:            {kr.get('mdd', 0):.2%}",
            f"  Trades:         {kr.get('n_trades', 0)}",
        ]

    # ── US ────────────────────────────────────────────────────────
    us = result.get("us", {})
    if us and "error" not in us:
        lines += [
            "",
            "── US Market (USD) ───────────────────────────────",
            f"  Total Return:   {us.get('total_return', 0):.2%}",
            f"  Sharpe:         {us.get('sharpe', 0):.3f}",
            f"  MDD:            {us.get('mdd', 0):.2%}",
            f"  Trades:         {us.get('n_trades', 0)}",
        ]

    # ── FX Impact ────────────────────────────────────────────────
    fx = result.get("fx_impact", {})
    if fx:
        lines += [
            "",
            "── FX Impact (USD/KRW) ───────────────────────────",
            f"  Rate Start:     {fx.get('usdkrw_start', 0):,.1f}",
            f"  Rate End:       {fx.get('usdkrw_end', 0):,.1f}",
            f"  FX Change:      {fx.get('usdkrw_change_pct', 0):.2%}",
            f"  FX Gain (KRW):  {fx.get('fx_gain_krw', 0):,.0f}",
        ]

    lines.append("")
    lines.append("═" * 60)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Advanced Performance Metrics
# ═══════════════════════════════════════════════════════════════════════
#
# These augment compute_metrics() with selection-bias-aware Sharpe variants
# and a Calmar / detailed drawdown helper used by the validation runner.
# They take a return Series directly (not the equity_df) so they can be
# called with bootstrap samples and per-fold slices.

def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.035,
) -> float:
    """Annualized Sharpe with risk-free adjustment (default 3.5% KOFR-ish)."""
    rets = returns.dropna()
    if len(rets) < 20 or rets.std() == 0:
        return 0.0
    excess = rets - risk_free_rate / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.035,
) -> float:
    """Sortino: only penalizes downside volatility."""
    rets = returns.dropna()
    if len(rets) < 20:
        return 0.0
    excess = rets - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(
        np.sqrt(periods_per_year) * excess.mean() / downside.std()
    )


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """CAGR / |Max DD|."""
    rets = returns.dropna()
    if len(rets) < 20:
        return 0.0
    cum = (1 + rets).cumprod()
    cagr = cum.iloc[-1] ** (periods_per_year / len(rets)) - 1
    dd = (cum / cum.cummax() - 1).min()
    if abs(dd) < 1e-6:
        return 0.0
    return float(cagr / abs(dd))


def max_drawdown(returns: pd.Series) -> dict:
    """Detailed drawdown stats: depth, peak-to-trough duration, recovery."""
    rets = returns.dropna()
    if len(rets) == 0:
        return {"max_dd": 0.0, "duration_days": 0, "recovery_days": None}
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    max_dd = float(dd.min())

    trough_idx = dd.idxmin()
    peak_idx = cum.loc[:trough_idx].idxmax()
    duration = (
        (trough_idx - peak_idx).days
        if hasattr(trough_idx, "to_pydatetime") else 0
    )

    post_trough = cum.loc[trough_idx:]
    recovery_threshold = peak.loc[peak_idx]
    recovered = post_trough[post_trough >= recovery_threshold]
    recovery_days = (
        (recovered.index[0] - trough_idx).days
        if len(recovered) > 0 and hasattr(trough_idx, "to_pydatetime")
        else None
    )

    return {
        "max_dd": max_dd,
        "duration_days": duration,
        "recovery_days": recovery_days,
    }


def deflated_sharpe(
    sr_observed: float,
    n_trials: int,
    returns: pd.Series,
    sr_benchmark: float = 0.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Returns the probability that the observed Sharpe exceeds the
    benchmark Sharpe, conditional on having tried n_trials variants.
    Corrects for selection bias and non-normal returns (skew/kurtosis).

    DSR > 0.95 is the minimum statistical bar; > 0.99 is "robust".
    """
    from scipy import stats

    rets = returns.dropna()
    n = len(rets)
    if n < 30:
        return 0.0

    emc = 0.5772156649  # Euler-Mascheroni
    if n_trials <= 1:
        sr_threshold = sr_benchmark
    else:
        sr_threshold = (
            (1 - emc) * stats.norm.ppf(1 - 1.0 / n_trials)
            + emc * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        )

    skew = float(rets.skew())
    kurt = float(rets.kurtosis())

    se_sharpe = np.sqrt(
        (1 - skew * sr_observed + (kurt / 4) * sr_observed ** 2) / (n - 1)
    )
    if not np.isfinite(se_sharpe) or se_sharpe <= 0:
        return 0.0

    z = (sr_observed - sr_threshold) / se_sharpe
    return float(stats.norm.cdf(z))


def probabilistic_sharpe(
    sr_observed: float,
    returns: pd.Series,
    sr_benchmark: float = 0.0,
) -> float:
    """
    PSR: probability the true Sharpe exceeds benchmark, accounting for
    non-normality. Equivalent to Deflated Sharpe with n_trials=1.
    """
    return deflated_sharpe(
        sr_observed, n_trials=1, returns=returns, sr_benchmark=sr_benchmark,
    )
