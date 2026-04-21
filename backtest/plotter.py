"""
Backtest Result Visualizations.

Generates four charts:
  1. Cumulative return: strategy vs KOSPI benchmark
  2. Drawdown curve
  3. Monthly returns heatmap
  4. Feature importance bar chart (top 20)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from backtest.metrics import monthly_returns

logger = logging.getLogger(__name__)

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")


def plot_all(
    equity_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    save_dir: str = "backtest_results",
) -> None:
    """
    Generate and save all backtest charts.

    Parameters
    ----------
    equity_df : Daily equity log from simulator.
    trade_df : Completed trade log from simulator.
    feature_importance : Averaged feature importance across folds.
    benchmark_df : KOSPI daily close (optional).
    save_dir : Directory to save PNG files.
    """
    from pathlib import Path
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_cumulative_return(equity_df, benchmark_df, out / "cumulative_return.png")
    plot_drawdown(equity_df, out / "drawdown.png")
    plot_monthly_heatmap(equity_df, out / "monthly_heatmap.png")
    if len(feature_importance) > 0:
        plot_feature_importance(feature_importance, out / "feature_importance.png")

    logger.info("All charts saved to %s/", save_dir)


# ═══════════════════════════════════════════════════════════════════════
# 1. Cumulative Return
# ═══════════════════════════════════════════════════════════════════════

def plot_cumulative_return(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None,
    save_path: str | None = None,
) -> None:
    """Plot cumulative return curve: strategy vs benchmark."""
    equity = equity_df.set_index("timestamp")["equity"].sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    cum_ret = equity / equity.iloc[0] - 1

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(cum_ret.index, cum_ret.values * 100, label="Strategy", linewidth=1.5)

    if benchmark_df is not None:
        bm = benchmark_df["close"].sort_index()
        bm = bm.reindex(cum_ret.index, method="ffill").dropna()
        if len(bm) > 1:
            bm_ret = bm / bm.iloc[0] - 1
            ax.plot(bm_ret.index, bm_ret.values * 100,
                    label="KOSPI", linewidth=1.2, alpha=0.7)

    ax.set_title("Cumulative Return", fontsize=14, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# 2. Drawdown Curve
# ═══════════════════════════════════════════════════════════════════════

def plot_drawdown(
    equity_df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Plot underwater (drawdown) curve."""
    equity = equity_df.set_index("timestamp")["equity"].sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    cum_max = equity.cummax()
    drawdown = (equity - cum_max) / cum_max * 100

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color="crimson", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="crimson",
            linewidth=0.8, alpha=0.8)

    ax.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# 3. Monthly Returns Heatmap
# ═══════════════════════════════════════════════════════════════════════

def plot_monthly_heatmap(
    equity_df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Plot monthly returns as a color-coded heatmap."""
    table = monthly_returns(equity_df)
    if table.empty:
        logger.warning("No monthly data for heatmap.")
        return

    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    fig, ax = plt.subplots(figsize=(12, max(3, len(table) * 0.6)))
    sns.heatmap(
        table.astype(float),
        annot=True, fmt=".1f", center=0,
        cmap="RdYlGn", linewidths=0.5,
        xticklabels=month_labels,
        yticklabels=table.index,
        ax=ax,
        cbar_kws={"label": "Return (%)"},
    )
    ax.set_title("Monthly Returns (%)", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# 4. Feature Importance
# ═══════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_path: str | None = None,
    top_n: int = 20,
) -> None:
    """Plot top N features by importance (gain)."""
    top = importance_df.head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    ax.barh(top["feature"], top["importance"], color="steelblue", alpha=0.85)
    ax.set_title(f"Top {top_n} Feature Importance (Gain)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Gain")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
