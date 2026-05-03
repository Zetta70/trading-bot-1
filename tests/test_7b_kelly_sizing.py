"""
Patch 7b — live Kelly sizing in TradingBot._execute().

Covers test cases 5-12 from the Patch-7 spec (R8):

  5.  _size_buy_kelly() returns None when portfolio is None.
  6.  _size_buy_kelly() returns None when ML prob is the default 0.5.
  7.  _size_buy_kelly() returns int >= 1 when all inputs valid.
  8.  Kelly is capped by available cash even when Kelly says higher.
  9.  End-to-end: Kelly path produces a Kelly-sized BUY; toggling
      USE_KELLY_SIZING=false produces the cash-only size.
  10. Patch 2 backward-compat: bot without portfolio uses cash-only
      sizing without warnings.
  11. Patch 5 lock discipline: Kelly computation runs OUTSIDE the
      cash-mutation lock (source-level guard).
  12. Source-level guard: every BUY skip path in _execute() logs a
      reason.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_predictor import MLPredictor  # noqa: E402
from portfolio_manager import PortfolioManager  # noqa: E402
from trading_bot import TradingBot  # noqa: E402


# ── Fakes / helpers ───────────────────────────────────────────────────

def _make_ohlcv(n=80, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.05, 0.6, n))
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.1, n),
        "high": close + np.abs(rng.normal(0.5, 0.2, n)),
        "low":  close - np.abs(rng.normal(0.5, 0.2, n)),
        "close": close,
        "volume": rng.integers(100_000, 500_000, n),
    }, index=dates)


class _FakeUSClient:
    def __init__(self, fx=1380.0):
        self._cached_usdkrw = fx
        self.default_usdkrw = 1380.0

    async def place_order(self, ticker, qty, side, order_type="market", price=0):
        return {"output": {"ODNO": "ORD"}}

    async def get_order_fill(self, _):
        return {"avg_fill_price": 0, "filled_qty": 0}

    async def get_current_price(self, _):
        return 130.0

    async def get_us_price(self, _):
        return 130.0


class _FakeScanner:
    async def scan(self): return []


class _FakeKRClientForPM:
    async def get_balance(self):
        return {"deposit": 1_000_000, "total_eval": 0, "holdings": []}

    async def get_current_price(self, _):
        return 70_000


def _new_predictor_with_prob(prob=0.62):
    """Return an MLPredictor whose ML model fixes a known probability."""
    p = MLPredictor(threshold_buy=0.55, threshold_sell=0.45)

    class _Model:
        feature_names: list[str] = []

        def predict_proba(self, X):
            return np.array([prob] * len(X))

    p._ml_model = _Model()
    return p


def _new_pm():
    return PortfolioManager(
        client=_FakeKRClientForPM(),
        scanner=_FakeScanner(),
        predictor=MLPredictor(),
        initial_cash=1_000_000,
        max_drawdown_pct=0.03,
        max_active_bots=5,
    )


def _new_bot_under_pm(
    pm,
    *,
    cash=200_000,
    fx=1380.0,
    prob=0.62,
    use_kelly=True,
    kelly_max_weight=0.10,
    daily_ohlcv=None,
):
    """Construct a TradingBot wired into ``pm`` with a fixed-prob model."""
    predictor = _new_predictor_with_prob(prob=prob)
    bot = TradingBot(
        client=_FakeUSClient(fx=fx),
        ticker="NVDA",
        predictor=predictor,
        currency="USD",
        initial_cash=cash,
        portfolio=pm,
        cash_lock=pm._cash_lock,
        use_kelly_sizing=use_kelly,
        kelly_multiplier=0.25,
        target_portfolio_vol=0.15,
        kelly_max_weight=kelly_max_weight,
    )
    bot.cash = cash
    bot.position = 0
    bot._daily_ohlcv = daily_ohlcv if daily_ohlcv is not None else _make_ohlcv()
    # Pre-seed the ML cache so the bot doesn't try to refresh from the
    # OHLCV cache (which we haven't stubbed for these tests).
    bot._cached_ml_signal = "BUY"
    bot._cached_ml_prob = float(prob)
    bot._last_ml_check = 1e18  # far future = TTL never expires here
    return bot


# ── R8 #5 — None when portfolio is None ───────────────────────────────

def test_kelly_returns_none_without_portfolio():
    pred = _new_predictor_with_prob(0.62)
    bot = TradingBot(
        client=_FakeUSClient(),
        ticker="NVDA",
        predictor=pred,
        currency="USD",
        initial_cash=200_000,
        # NO portfolio kwarg → must fall back to cash-only sizing
    )
    bot._daily_ohlcv = _make_ohlcv()
    bot._cached_ml_prob = 0.62
    bot._last_ml_check = 1e18
    assert bot._size_buy_kelly(price=130.0, fx=1380.0) is None


# ── R8 #6 — None when ML prob is the default 0.5 sentinel ─────────────

def test_kelly_returns_none_when_prob_is_05_sentinel():
    pm = _new_pm()
    bot = _new_bot_under_pm(pm, prob=0.5)
    # Override cache to default sentinel.
    bot._cached_ml_prob = 0.5
    assert bot._size_buy_kelly(price=130.0, fx=1380.0) is None


# ── R8 #7 — Returns int >= 1 with all inputs valid ────────────────────

def test_kelly_returns_positive_int_with_valid_inputs():
    pm = _new_pm()
    # 10% of total equity must cover at least one share. With NVDA $130
    # at fx 1380, one share ≈ 180k KRW; 10% of 5M = 500k KRW comfortably
    # affords a 1+ share Kelly position.
    pm._remaining_cash = 5_000_000
    bot = _new_bot_under_pm(pm, cash=2_000_000, prob=0.62)
    qty = bot._size_buy_kelly(price=130.0, fx=1380.0)
    assert qty is not None, "Kelly returned None despite valid inputs"
    assert isinstance(qty, int)
    assert qty >= 1


# ── R8 #8 — Cash cap dominates Kelly upper bound ──────────────────────

def test_kelly_capped_by_available_cash():
    """Kelly says huge size (large equity, big edge) but bot.cash is
    a small slice — qty must respect the cash cap."""
    pm = _new_pm()
    # Whisper a 10M equity by setting remaining_cash way larger than the
    # bot's slice. Total_equity will reflect this through the property.
    pm._remaining_cash = 10_000_000
    bot = _new_bot_under_pm(
        pm, cash=200_000, prob=0.85,  # high conviction → bigger Kelly
        kelly_max_weight=0.50,        # let Kelly want a lot
    )
    qty = bot._size_buy_kelly(price=130.0, fx=1380.0)
    assert qty is not None

    cps_krw = bot._cost_per_share_krw(130.0, 1380.0)
    max_qty_by_cash = int(bot.cash * 0.95 // cps_krw)
    assert qty <= max_qty_by_cash, (
        f"Kelly qty {qty} exceeds cash-cap {max_qty_by_cash}"
    )


# ── R8 #9 — End-to-end: Kelly vs cash-only divergence ─────────────────

def test_kelly_path_vs_cash_only_diverge_under_low_conviction():
    """Same bot, same cash, same price — toggling USE_KELLY_SIZING
    should yield a Kelly qty <= cash-only qty when prob is modest
    (Kelly fraction at p≈0.55 with 0.25 multiplier is well below the
    10% max weight, so it sizes smaller than 'all available cash')."""
    # Kelly ON
    pm = _new_pm()
    pm._remaining_cash = 1_000_000
    bot_k = _new_bot_under_pm(pm, cash=500_000, prob=0.55)

    async def go_kelly():
        await bot_k._execute("BUY", 130.0)
    asyncio.run(go_kelly())
    kelly_position = bot_k.position
    assert kelly_position >= 0

    # Kelly OFF
    pm2 = _new_pm()
    pm2._remaining_cash = 1_000_000
    bot_c = _new_bot_under_pm(
        pm2, cash=500_000, prob=0.55, use_kelly=False,
    )

    async def go_cash():
        await bot_c._execute("BUY", 130.0)
    asyncio.run(go_cash())
    cash_position = bot_c.position
    assert cash_position >= 1

    assert kelly_position <= cash_position, (
        f"Kelly produced larger position ({kelly_position}) than "
        f"cash-only ({cash_position}) at modest conviction — "
        f"shouldn't happen with kelly_multiplier=0.25"
    )


# ── R8 #10 — Patch 2 backward compat ──────────────────────────────────

def test_bot_without_portfolio_uses_cash_only_no_warnings(caplog):
    """No portfolio kwarg, no Kelly config — Patch 2 path runs cleanly."""
    pred = MLPredictor()
    pred._ml_model = None  # legacy path
    bot = TradingBot(
        client=_FakeUSClient(),
        ticker="NVDA",
        predictor=pred,
        currency="USD",
        initial_cash=200_000,
    )
    bot.cash = 200_000
    bot.position = 0

    with caplog.at_level(logging.WARNING):
        asyncio.run(bot._execute("BUY", 130.0))

    # Cash-only sized: 200k * 0.95 / (130 * 1.00115 * 1380) ≈ 1.06 → 1
    assert bot.position == 1
    # No WARNINGs/ERRORs — the missing portfolio is a normal fallback.
    bad = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert bad == [], (
        f"Backward-compat path emitted unexpected warnings: "
        f"{[r.message for r in bad]}"
    )


# ── R8 #11 — Lock discipline: Kelly runs OUTSIDE the cash lock ────────

def test_kelly_computation_runs_outside_cash_lock():
    """Source-level structural guard: in _execute(), the call to
    _size_buy_kelly must come BEFORE the `async with self._cash_lock_ctx():`
    block. Holding the lock across Kelly (which reads other bots'
    OHLCV, computes vol, etc.) would serialize all bots."""
    src = (Path(__file__).resolve().parent.parent / "trading_bot.py").read_text()

    # Find the start of _execute().
    exec_start = src.index("async def _execute(self, side: str, price)")
    body = src[exec_start:]

    kelly_idx = body.index("self._size_buy_kelly(")
    lock_idx = body.index("self._cash_lock_ctx()")
    assert kelly_idx < lock_idx, (
        "Kelly is called inside / after the cash lock context — "
        "this serializes all bots on the network await."
    )


# ── R8 #12 — Every BUY skip logs a reason ─────────────────────────────

def test_buy_skip_paths_log_reasons_in_execute():
    """Every `return` inside the BUY branch of _execute must be
    preceded by at least one logger.* call (info/warning) so live
    runs are debuggable."""
    src = (Path(__file__).resolve().parent.parent / "trading_bot.py").read_text()
    exec_start = src.index("async def _execute(self, side: str, price)")
    sell_branch = src.index("else:  # SELL", exec_start)
    buy_body = src[exec_start:sell_branch]

    # Pull every "return$" (no value, early-exit) inside the BUY branch.
    return_lines = [
        m.start() for m in re.finditer(r"\n\s+return\s*$", buy_body, re.M)
    ]
    assert len(return_lines) >= 3, (
        "Expected >=3 early-return paths in BUY branch, found "
        f"{len(return_lines)}"
    )

    for pos in return_lines:
        preceding = buy_body[:pos]
        # Look for a logger call within the preceding ~6 lines.
        last_lines = preceding.rsplit("\n", 7)[-7:]
        snippet = "\n".join(last_lines)
        assert "logger." in snippet, (
            f"BUY early-return at offset {pos} has no logger call "
            f"in the preceding 6 lines:\n{snippet}"
        )


# ── Logging shape (R7) ────────────────────────────────────────────────

def test_kelly_log_includes_debug_dict_fields(caplog):
    pm = _new_pm()
    pm._remaining_cash = 1_000_000
    bot = _new_bot_under_pm(pm, cash=500_000, prob=0.62)

    with caplog.at_level(logging.INFO):
        bot._size_buy_kelly(130.0, 1380.0)

    msgs = [r.message for r in caplog.records]
    kelly_logs = [m for m in msgs if "BUY sizing: kelly path" in m]
    assert kelly_logs, "Expected a 'kelly path' INFO log line"
    line = kelly_logs[-1]
    for needle in (
        "prob=", "vol=", "kelly_w=", "vol_w=", "corr_mult=",
        "final_w=", "amount=", "qty=",
    ):
        assert needle in line, f"Missing {needle!r} in kelly log: {line}"
