"""
Patch 7a — probability flow + portfolio returns dict.

Covers test cases 1-4 from the Patch-7 spec (R8):

  1. predict_signal_with_prob() returns the model's real probability
     when a model is loaded; falls back to (signal, score-as-prob)
     when model is None.
  2. _get_ml_signal_with_prob() honors the 1-hour TTL.
  3. _get_ml_signal() still works unchanged (regression).
  4. PortfolioManager.get_returns_dict() excludes the candidate ticker
     and skips bots with insufficient OHLCV.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_predictor import MLPredictor  # noqa: E402
from portfolio_manager import PortfolioManager  # noqa: E402
from trading_bot import TradingBot  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 80, seed: int = 0) -> pd.DataFrame:
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


class FixedProbModel:
    """Mimics StockPredictor for predict_proba; returns a fixed value."""
    def __init__(self, fixed_prob=0.62):
        self.fixed_prob = fixed_prob
        self.feature_names: list[str] = []

    def predict_proba(self, X):
        return np.array([self.fixed_prob] * len(X))


# ── R8 #1 — predict_signal_with_prob ──────────────────────────────────

def test_predict_signal_with_prob_returns_real_prob_when_model_loaded():
    p = MLPredictor(threshold_buy=0.55, threshold_sell=0.45)
    p._ml_model = FixedProbModel(fixed_prob=0.62)
    df = _make_ohlcv()
    sig, prob = p.predict_signal_with_prob(df)
    assert sig == "BUY"
    assert prob == pytest.approx(0.62, abs=1e-9)


def test_predict_signal_with_prob_falls_back_when_no_model():
    """No model → legacy bullish score is returned as the probability."""
    p = MLPredictor(threshold_buy=0.55, threshold_sell=0.45)
    p._ml_model = None
    df = _make_ohlcv()
    sig, prob = p.predict_signal_with_prob(df)
    assert sig in ("BUY", "SELL", "HOLD")
    assert 0.0 <= prob <= 1.0


def test_predict_signal_with_prob_returns_05_on_nan_features():
    p = MLPredictor(threshold_buy=0.55, threshold_sell=0.45)
    p._ml_model = FixedProbModel(fixed_prob=0.62)
    # Tiny OHLCV → not enough rows for rolling features → NaN row.
    df = _make_ohlcv(n=20)
    sig, prob = p.predict_signal_with_prob(df)
    assert sig == "HOLD"
    assert prob == 0.5


def test_predict_signal_with_prob_never_raises():
    """Spec: never raises; on any internal error falls back gracefully."""
    p = MLPredictor()
    class Boom:
        feature_names = []
        def predict_proba(self, X): raise RuntimeError("boom")
    p._ml_model = Boom()
    df = _make_ohlcv()
    sig, prob = p.predict_signal_with_prob(df)  # must not raise
    assert sig in ("BUY", "SELL", "HOLD")
    assert 0.0 <= prob <= 1.0


def test_predict_signal_remains_a_thin_wrapper():
    """predict_signal must return ONLY the signal string (no tuple).
    Public API is the regression target."""
    p = MLPredictor(threshold_buy=0.55, threshold_sell=0.45)
    p._ml_model = FixedProbModel(fixed_prob=0.62)
    df = _make_ohlcv()
    out = p.predict_signal(df)
    assert isinstance(out, str)
    assert out == "BUY"


# ── R8 #2,3 — TradingBot cache + TTL + signal-only getter ─────────────

class _FakeClient:
    async def get_current_price(self, _): return 100
    async def place_order(self, *a, **kw): return {"output": {"ODNO": "X"}}
    async def get_order_fill(self, _): return {"avg_fill_price": 0, "filled_qty": 0}


def _new_bot_with_predictor(predictor):
    return TradingBot(
        client=_FakeClient(),
        ticker="NVDA",
        predictor=predictor,
        currency="USD",
        initial_cash=200_000,
    )


def test_cache_holds_signal_and_prob_after_first_refresh(monkeypatch):
    p = MLPredictor()
    p._ml_model = FixedProbModel(fixed_prob=0.62)
    bot = _new_bot_with_predictor(p)
    df = _make_ohlcv()

    # Make the OHLCV cache return our synthetic frame.
    monkeypatch.setattr(bot._ohlcv_cache, "get", lambda _t: df)

    sig, prob = bot._get_ml_signal_with_prob()
    assert sig == "BUY"
    assert prob == pytest.approx(0.62, abs=1e-9)
    # Cache fields should also be populated.
    assert bot._cached_ml_signal == "BUY"
    assert bot._cached_ml_prob == pytest.approx(0.62, abs=1e-9)


def test_get_ml_signal_with_prob_honors_ttl(monkeypatch):
    """Within TTL the predictor must NOT be called again."""
    call_count = {"n": 0}

    class CountingModel(FixedProbModel):
        def predict_proba(self, X):
            call_count["n"] += 1
            return super().predict_proba(X)

    p = MLPredictor()
    p._ml_model = CountingModel(fixed_prob=0.62)
    bot = _new_bot_with_predictor(p)
    df = _make_ohlcv()
    monkeypatch.setattr(bot._ohlcv_cache, "get", lambda _t: df)

    bot._get_ml_signal_with_prob()
    bot._get_ml_signal_with_prob()
    bot._get_ml_signal_with_prob()
    assert call_count["n"] == 1, (
        "TTL broken — predictor was re-called within ML_CHECK_INTERVAL"
    )

    # Force expiry by rewinding _last_ml_check past the TTL.
    bot._last_ml_check = time.time() - bot.ML_CHECK_INTERVAL - 1
    bot._get_ml_signal_with_prob()
    assert call_count["n"] == 2


def test_get_ml_signal_signature_unchanged(monkeypatch):
    """Regression: _get_ml_signal returns a str, not a tuple."""
    p = MLPredictor()
    p._ml_model = FixedProbModel(fixed_prob=0.62)
    bot = _new_bot_with_predictor(p)
    df = _make_ohlcv()
    monkeypatch.setattr(bot._ohlcv_cache, "get", lambda _t: df)

    out = bot._get_ml_signal()
    assert isinstance(out, str)
    assert out in ("BUY", "SELL", "HOLD")


# ── R8 #4 — PortfolioManager.get_returns_dict ─────────────────────────

class _FakeScanner:
    async def scan(self): return []


class _FakePredictor:
    _ml_model = None
    def score(self, _): return 0.5


def _new_pm():
    return PortfolioManager(
        client=_FakeClient(),
        scanner=_FakeScanner(),
        predictor=_FakePredictor(),
        initial_cash=1_000_000,
        max_drawdown_pct=0.03,
        max_active_bots=5,
    )


def _stub_bot(ticker: str, ohlcv: pd.DataFrame | None) -> TradingBot:
    bot = TradingBot(
        client=_FakeClient(),
        ticker=ticker,
        predictor=_FakePredictor(),
        currency="USD",
        initial_cash=200_000,
    )
    bot._daily_ohlcv = ohlcv
    return bot


def test_returns_dict_excludes_candidate_ticker():
    pm = _new_pm()
    df = _make_ohlcv(n=80, seed=1)
    pm.bots["NVDA"] = _stub_bot("NVDA", df)
    pm.bots["AAPL"] = _stub_bot("AAPL", _make_ohlcv(n=80, seed=2))

    out = pm.get_returns_dict(exclude_ticker="NVDA")
    assert "NVDA" not in out
    assert "AAPL" in out
    assert isinstance(out["AAPL"], pd.Series)
    assert len(out["AAPL"]) > 0


def test_returns_dict_skips_bots_with_no_ohlcv():
    pm = _new_pm()
    pm.bots["NVDA"] = _stub_bot("NVDA", None)  # no OHLCV
    pm.bots["AAPL"] = _stub_bot("AAPL", _make_ohlcv(n=80, seed=3))

    out = pm.get_returns_dict()
    assert "NVDA" not in out
    assert "AAPL" in out


def test_returns_dict_skips_bots_with_insufficient_rows():
    pm = _new_pm()
    pm.bots["NVDA"] = _stub_bot("NVDA", _make_ohlcv(n=10, seed=4))  # < 30
    pm.bots["AAPL"] = _stub_bot("AAPL", _make_ohlcv(n=80, seed=5))

    out = pm.get_returns_dict()
    assert "NVDA" not in out
    assert "AAPL" in out


def test_returns_dict_lookback_caps_series_length():
    pm = _new_pm()
    pm.bots["AAPL"] = _stub_bot("AAPL", _make_ohlcv(n=200, seed=6))
    out = pm.get_returns_dict(lookback=40)
    assert len(out["AAPL"]) == 40
