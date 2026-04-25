"""
Integrated Validation Runner.

Usage
-----
    python -m backtest.validate --model models/lgbm_v3_meta.pkl

Runs the Phase 4 validation battery and prints a PASS/FAIL report. The
JSON report is also written to ``validation_results/validation_report.json``.

The validation period must NOT overlap the training period — the
``--train-end`` / ``--validate-start`` flags enforce that.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path when run as ``python -m``.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


def run_validation(
    model_path: str,
    train_end: str = "2024-06-30",
    validate_start: str = "2024-07-01",
    validate_end: str = "2024-12-31",
    n_trials: int = 50,
    output_dir: str = "validation_results",
) -> dict:
    """Execute the full validation battery and return the report dict."""
    from config import Config
    from backtest.data_loader import load_ohlcv, load_kospi, load_vkospi
    from backtest.simulator import TradingSimulator
    from backtest.walk_forward import WalkForwardBacktest
    from backtest.metrics import (
        sharpe_ratio, sortino_ratio, calmar_ratio,
        deflated_sharpe, probabilistic_sharpe, max_drawdown,
    )
    from backtest.monte_carlo import trade_shuffle_mc, block_bootstrap_mc
    from backtest.walk_forward_analysis import fold_consistency_report
    from backtest.stress_test import run_stress_test, STRESS_WINDOWS_KR
    from backtest.trade_analytics import analyze_trades

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cfg = Config()

    report: dict = {
        "model_path": model_path,
        "train_end": train_end,
        "validate_start": validate_start,
        "validate_end": validate_end,
        "n_trials_assumed": n_trials,
    }

    # ── 1. Out-of-sample backtest ──────────────────────────────
    logger.info("Running OOS backtest: %s ~ %s", validate_start, validate_end)
    tickers = cfg.tickers_kr or cfg.tickers

    ticker_ohlcv: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = load_ohlcv(t, validate_start, validate_end)
            if len(df) >= 30:
                ticker_ohlcv[t] = df
        except Exception as e:
            logger.warning("Load failed for %s: %s", t, e)

    if not ticker_ohlcv:
        return {**report, "error": "no OOS data loadable"}

    try:
        index_df = load_kospi(validate_start, validate_end)
    except Exception as e:
        logger.warning("KOSPI load failed: %s", e)
        index_df = None

    vkospi_df = None
    if cfg.use_vol_index:
        try:
            vk = load_vkospi(validate_start, validate_end)
            vkospi_df = vk if not vk.empty else None
        except Exception:
            vkospi_df = None

    sim = TradingSimulator(
        initial_capital=cfg.initial_cash,
        commission=0.00015,
        tax=0.0023,
        slippage=0.001,
        use_kelly_sizing=cfg.use_kelly_sizing,
        kelly_multiplier=cfg.kelly_multiplier,
        pt_atr_mult=cfg.bt_pt_atr_mult,
        sl_atr_mult=cfg.bt_sl_atr_mult,
    )
    wf = WalkForwardBacktest(
        train_window=max(cfg.bt_train_window, 63),
        test_window=cfg.bt_test_window,
        horizon=cfg.bt_horizon,
        pt_atr_mult=cfg.bt_pt_atr_mult,
        sl_atr_mult=cfg.bt_sl_atr_mult,
        use_ensemble=cfg.use_ensemble,
        use_meta_labeling=cfg.use_meta_labeling,
    )
    result = wf.run(ticker_ohlcv, index_df, simulator=sim, vol_index_df=vkospi_df)
    sim = result["simulator"]

    # ── 2. Core metrics ────────────────────────────────────────
    equity = pd.Series([e["equity"] for e in sim.daily_equity])
    if len(equity) < 10:
        return {**report, "error": "backtest too short"}

    returns = equity.pct_change().dropna()
    trade_log = sim.get_trade_log()

    sr = sharpe_ratio(returns)
    sor = sortino_ratio(returns)
    cal = calmar_ratio(returns)
    dd_info = max_drawdown(returns)
    dsr = deflated_sharpe(sr, n_trials=n_trials, returns=returns)
    psr = probabilistic_sharpe(sr, returns)

    report["oos_metrics"] = {
        "n_days": len(returns),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "sharpe": sr,
        "sortino": sor,
        "calmar": cal,
        "max_dd": dd_info["max_dd"],
        "dd_duration_days": dd_info["duration_days"],
        "recovery_days": dd_info["recovery_days"],
        "deflated_sharpe": dsr,
        "probabilistic_sharpe": psr,
        "n_trades": len(trade_log),
    }

    # ── 3. Monte Carlo ─────────────────────────────────────────
    if not trade_log.empty:
        report["trade_shuffle_mc"] = trade_shuffle_mc(trade_log["return_pct"])
    report["block_bootstrap_mc"] = block_bootstrap_mc(returns)

    # ── 4. Fold consistency ────────────────────────────────────
    equity_df = pd.DataFrame(sim.daily_equity)
    report["fold_consistency"] = fold_consistency_report(
        result["fold_results"], equity_df,
    )

    # ── 5. Stress test ─────────────────────────────────────────
    try:
        ticker_ohlcv_full: dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                df = load_ohlcv(t, "2019-01-01", validate_end)
                if len(df) >= 30:
                    ticker_ohlcv_full[t] = df
            except Exception:
                pass
        try:
            index_full = load_kospi("2019-01-01", validate_end)
        except Exception:
            index_full = None
        report["stress_test"] = run_stress_test(
            wf, ticker_ohlcv_full, index_full, STRESS_WINDOWS_KR,
        )
    except Exception as e:
        logger.warning("Stress test skipped: %s", e)
        report["stress_test"] = {"error": str(e)}

    # ── 6. Trade analytics ─────────────────────────────────────
    if not trade_log.empty:
        report["trade_analytics"] = analyze_trades(
            trade_log, equity_df, vkospi_df,
        )

    # ── 7. PASS/FAIL gates ─────────────────────────────────────
    decisions = evaluate_decision(report)
    report["decisions"] = decisions
    report["overall_verdict"] = (
        "PASS" if all(d["pass"] for d in decisions.values()) else "FAIL"
    )

    out_path = Path(output_dir) / "validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Validation report saved to %s", out_path)

    return report


def evaluate_decision(report: dict) -> dict:
    """
    Hard gates that must ALL pass before live capital deployment.

    Conservative thresholds — err toward rejection. Tweaking these is
    a strategic decision (your risk tolerance), not a tuning knob.
    """
    m = report.get("oos_metrics", {})
    mc_block = report.get("block_bootstrap_mc", {})
    fold = report.get("fold_consistency", {})
    stress = report.get("stress_test", [])
    analytics = report.get("trade_analytics", {})

    stress_dds = [
        s.get("max_dd", 0)
        for s in stress
        if isinstance(s, dict) and "max_dd" in s
    ]

    return {
        "deflated_sharpe_gte_0.95": {
            "value": m.get("deflated_sharpe", 0),
            "threshold": 0.95,
            "pass": m.get("deflated_sharpe", 0) >= 0.95,
            "reason": "Selection-bias-corrected edge probability",
        },
        "oos_sharpe_gte_1.0": {
            "value": m.get("sharpe", 0),
            "threshold": 1.0,
            "pass": m.get("sharpe", 0) >= 1.0,
            "reason": "Out-of-sample Sharpe minimum for live trading",
        },
        "max_dd_within_15pct": {
            "value": m.get("max_dd", 0),
            "threshold": -0.15,
            "pass": m.get("max_dd", 0) >= -0.15,
            "reason": "OOS drawdown tolerable for personal capital",
        },
        "block_bootstrap_p_lt_0.05": {
            "value": mc_block.get("p_value", 1.0),
            "threshold": 0.05,
            "pass": mc_block.get("p_value", 1.0) < 0.05,
            "reason": "Sharpe must beat bootstrap null at 95% confidence",
        },
        "fold_stability_gte_0.70": {
            "value": fold.get("stability_score", 0),
            "threshold": 0.70,
            "pass": fold.get("stability_score", 0) >= 0.70,
            "reason": "≥70% of folds must have positive Sharpe",
        },
        "no_stress_catastrophe": {
            "value": min(stress_dds, default=0),
            "threshold": -0.25,
            "pass": all(dd >= -0.25 for dd in stress_dds),
            "reason": "No crisis window produced >25% drawdown",
        },
        "min_n_trades_gte_50": {
            "value": m.get("n_trades", 0),
            "threshold": 50,
            "pass": m.get("n_trades", 0) >= 50,
            "reason": "≥50 trades required for statistical confidence",
        },
        "profit_factor_gte_1.3": {
            "value": analytics.get("profit_factor", 0),
            "threshold": 1.3,
            "pass": analytics.get("profit_factor", 0) >= 1.3,
            "reason": "Winners must materially exceed losers",
        },
    }


def print_report(report: dict) -> None:
    """Pretty-print the verdict table."""
    print("\n" + "=" * 70)
    print("  VALIDATION REPORT")
    print("=" * 70)
    print(f"  Verdict     : {report.get('overall_verdict', 'UNKNOWN')}")
    print(
        f"  OOS Period  : {report.get('validate_start')} ~ "
        f"{report.get('validate_end')}"
    )
    print("=" * 70)

    m = report.get("oos_metrics", {})
    if m:
        print(f"  OOS Sharpe       : {m.get('sharpe', 0):.3f}")
        print(f"  Deflated Sharpe  : {m.get('deflated_sharpe', 0):.3f}")
        print(f"  Max Drawdown     : {m.get('max_dd', 0) * 100:.2f}%")
        print(f"  Trades           : {m.get('n_trades', 0)}")
    print("-" * 70)

    for name, d in report.get("decisions", {}).items():
        status = "PASS" if d["pass"] else "FAIL"
        print(
            f"  [{status}]  {name:<35} value={d['value']}"
        )
        if not d["pass"]:
            print(f"            -> {d['reason']}")
    print("=" * 70)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path to trained model",
    )
    parser.add_argument("--train-end", default="2024-06-30")
    parser.add_argument("--validate-start", default="2024-07-01")
    parser.add_argument("--validate-end", default="2024-12-31")
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of strategy variants tried (for DSR)",
    )
    parser.add_argument("--output", default="validation_results")
    args = parser.parse_args()

    report = run_validation(
        model_path=args.model,
        train_end=args.train_end,
        validate_start=args.validate_start,
        validate_end=args.validate_end,
        n_trials=args.n_trials,
        output_dir=args.output,
    )
    print_report(report)

    if report.get("overall_verdict") != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
