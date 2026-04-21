"""
Multi-Ticker Equity Curve Dashboard.

Reads equity_log.csv and trade_log.csv and plots:
  - Per-ticker price charts with BUY/SELL markers
  - Portfolio-level equity curve with P&L shading
  - Per-ticker equity breakdown

Usage:
    python plot_equity.py              # One-shot PNG
    python plot_equity.py --live       # Auto-refresh every 30s
    python plot_equity.py --ticker 005930  # Single ticker view
"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

EQUITY_LOG = "equity_log.csv"
TRADE_LOG = "trade_log.csv"


def read_equity_log() -> dict[str, list[tuple[datetime, int, int]]]:
    """Return {ticker: [(timestamp, price, equity), ...]}."""
    data: dict[str, list] = defaultdict(list)
    path = Path(EQUITY_LOG)
    if not path.exists():
        return data

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            ticker = row["ticker"]
            price = int(row["price"]) if row["price"] else 0
            equity = int(row["equity"])
            data[ticker].append((ts, price, equity))

    return data


def read_trade_log() -> list[dict]:
    """Return list of trade dicts."""
    trades = []
    path = Path(TRADE_LOG)
    if not path.exists():
        return trades

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["timestamp"] = datetime.strptime(
                row["timestamp"], "%Y-%m-%d %H:%M:%S",
            )
            row["price"] = int(row["price"])
            trades.append(row)

    return trades


def plot(live: bool = False, ticker_filter: str | None = None) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Paper Trading Bot — Multi-Ticker Dashboard",
        fontsize=14, fontweight="bold",
    )

    def update():
        fig.clear()

        equity_data = read_equity_log()
        trades = read_trade_log()

        if not equity_data:
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "No data yet.\nRun the bot first: python main.py",
                ha="center", va="center", fontsize=14,
                transform=ax.transAxes,
            )
            return

        # Separate portfolio vs per-ticker data
        portfolio = equity_data.pop("PORTFOLIO", [])
        tickers = sorted(equity_data.keys())

        if ticker_filter:
            tickers = [t for t in tickers if t == ticker_filter]

        n_tickers = len(tickers)
        n_rows = n_tickers + 1  # +1 for portfolio

        # ── Portfolio equity (top) ───────────────────────────────
        ax_port = fig.add_subplot(n_rows, 1, 1)

        if portfolio:
            p_ts = [r[0] for r in portfolio]
            p_eq = [r[2] for r in portfolio]
            initial = p_eq[0]

            ax_port.plot(p_ts, p_eq, color="#FF9800", linewidth=1.5)
            ax_port.axhline(
                y=initial, color="gray", linestyle="--", linewidth=0.8,
            )
            ax_port.fill_between(
                p_ts, p_eq, initial,
                where=[e >= initial for e in p_eq],
                alpha=0.15, color="#4CAF50",
            )
            ax_port.fill_between(
                p_ts, p_eq, initial,
                where=[e < initial for e in p_eq],
                alpha=0.15, color="#F44336",
            )

            final = p_eq[-1]
            pnl = final - initial
            pnl_pct = (pnl / initial) * 100 if initial else 0
            color = "#4CAF50" if pnl >= 0 else "#F44336"
            ax_port.set_title(
                f"Portfolio Equity  |  P&L: {pnl:+,} KRW ({pnl_pct:+.2f}%)",
                color=color, fontweight="bold",
            )
        else:
            ax_port.set_title("Portfolio Equity (no data)")

        ax_port.set_ylabel("Equity (KRW)")
        ax_port.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        # ── Per-ticker charts ────────────────────────────────────
        colors = ["#2196F3", "#9C27B0", "#009688", "#E91E63",
                   "#FF5722", "#795548", "#607D8B", "#3F51B5"]

        for i, ticker in enumerate(tickers):
            ax = fig.add_subplot(n_rows, 1, i + 2)
            records = equity_data[ticker]
            ts = [r[0] for r in records]
            prices = [r[1] for r in records]
            c = colors[i % len(colors)]

            ax.plot(ts, prices, color=c, linewidth=1.0, label=f"{ticker} Price")

            # Trade markers
            t_trades = [t for t in trades if t["ticker"] == ticker]
            buy_ts = [t["timestamp"] for t in t_trades if t["side"] == "BUY"]
            buy_pr = [t["price"] for t in t_trades if t["side"] == "BUY"]
            sell_ts = [t["timestamp"] for t in t_trades if t["side"] == "SELL"]
            sell_pr = [t["price"] for t in t_trades if t["side"] == "SELL"]

            ax.scatter(buy_ts, buy_pr, marker="^", color="#4CAF50",
                       s=80, zorder=5, label="BUY")
            ax.scatter(sell_ts, sell_pr, marker="v", color="#F44336",
                       s=80, zorder=5, label="SELL")

            ax.set_ylabel(f"{ticker} (KRW)")
            ax.legend(loc="upper left", fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        fig.autofmt_xdate()

    if live:
        plt.ion()
        print("Live mode — refreshing every 30s. Close window to stop.")
        while True:
            update()
            plt.tight_layout()
            plt.draw()
            plt.pause(30)
    else:
        update()
        plt.tight_layout()
        plt.savefig("equity_curve.png", dpi=150, bbox_inches="tight")
        print("Saved to equity_curve.png")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Equity Curve Dashboard")
    parser.add_argument("--live", action="store_true",
                        help="Auto-refresh every 30s")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Filter to a single ticker")
    args = parser.parse_args()
    plot(live=args.live, ticker_filter=args.ticker)
