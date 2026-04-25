"""
Patch 1 — currency-awareness in TradingBot.

Verifies:
  * USD bot deducts KRW (USD * fx) from self.cash on BUY
  * USD bot adds KRW (USD * fx) to self.cash on SELL
  * KRW bot is unchanged (FX = 1.0)
  * equity property converts USD position MTM to KRW
  * _fx_to_krw() falls back to default_usdkrw when cache is 0/NaN
  * _USClientAdapter no longer multiplies USD prices by 100
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_bot import TradingBot  # noqa: E402


# ── Fakes ─────────────────────────────────────────────────────────────

class FakePredictor:
    def score(self, _prices):
        return 0.5


class FakeUSClient:
    """Mimics _USClientAdapter contract — USD prices, FX accessors."""

    def __init__(self, fx=1380.0):
        self._cached_usdkrw = fx
        self.default_usdkrw = 1380.0
        self._next_fill_price = None
        self.last_order = None

    async def place_order(self, ticker, qty, side, order_type="market", price=0):
        self.last_order = {
            "ticker": ticker, "qty": qty, "side": side,
            "order_type": order_type, "price": price,
        }
        return {"output": {"ODNO": "ORD-1"}}

    async def get_order_fill(self, order_no):
        if self._next_fill_price is None:
            return {"avg_fill_price": 0, "filled_qty": 0}
        return {
            "avg_fill_price": float(self._next_fill_price),
            "filled_qty": 1,
        }

    async def get_current_price(self, ticker):
        return 130.50

    # _USClientAdapter uses the underlying client's get_us_price API name.
    async def get_us_price(self, ticker):
        return 130.50


class FakeKRClient:
    """Bare KRW client, integer prices, no FX."""

    def __init__(self):
        self.last_order = None

    async def place_order(self, ticker, qty, side, order_type="market", price=0):
        self.last_order = {
            "ticker": ticker, "qty": qty, "side": side,
            "order_type": order_type, "price": price,
        }
        return {"output": {"ODNO": "ORD-KR"}}

    async def get_order_fill(self, order_no):
        return {"avg_fill_price": 0, "filled_qty": 0}

    async def get_current_price(self, ticker):
        return 70_000


# ── Patch-1 tests ─────────────────────────────────────────────────────

def _new_bot(currency, client, **kw):
    return TradingBot(
        client=client,
        ticker="NVDA" if currency == "USD" else "005930",
        predictor=FakePredictor(),
        currency=currency,
        initial_cash=200_000,
        qty=1,
        commission=0.00015,
        tax_kr=0.0023,
        tax_us=0.0,
        **kw,
    )


def test_currency_validation():
    with pytest.raises(ValueError):
        _new_bot("EUR", FakeUSClient())


def test_fx_to_krw_kr_returns_one():
    bot = _new_bot("KRW", FakeKRClient())
    assert bot._fx_to_krw() == 1.0


def test_fx_to_krw_uses_cached_when_valid():
    client = FakeUSClient(fx=1400.0)
    bot = _new_bot("USD", client)
    assert bot._fx_to_krw() == 1400.0


def test_fx_to_krw_falls_back_to_default_when_zero():
    client = FakeUSClient(fx=0.0)
    bot = _new_bot("USD", client)
    assert bot._fx_to_krw() == 1380.0


def test_fx_to_krw_falls_back_when_nan():
    client = FakeUSClient(fx=float("nan"))
    bot = _new_bot("USD", client)
    assert bot._fx_to_krw() == 1380.0


def test_equity_usd_converts_to_krw():
    client = FakeUSClient(fx=1380.0)
    bot = _new_bot("USD", client)
    bot.cash = 50_000
    bot.position = 2
    bot.last_price = 130.0
    expected = int(50_000 + 2 * 130.0 * 1380.0)
    assert bot.equity == expected


def test_equity_krw_unchanged():
    bot = _new_bot("KRW", FakeKRClient())
    bot.cash = 100_000
    bot.position = 5
    bot.last_price = 70_000
    assert bot.equity == 100_000 + 5 * 70_000


def test_buy_usd_deducts_krw_using_fx():
    """NVDA $130 * 1 share * 1380 ≈ 179_400 KRW deducted from cash."""
    client = FakeUSClient(fx=1380.0)
    bot = _new_bot("USD", client)
    bot.cash = 200_000
    bot.position = 0
    bot.qty = 1

    asyncio.run(bot._execute("BUY", 130.0))

    # Expected: 130 * 1 * 1.00015 * 1380 ≈ 179_426
    expected_cost = int(130.0 * 1 * (1 + 0.00015) * 1380.0)
    assert bot.cash == 200_000 - expected_cost
    assert bot.position == 1
    # Reasonable sanity bound: between ~179k and ~180k.
    assert 179_000 < (200_000 - bot.cash) < 180_000


def test_sell_usd_adds_krw_using_fx():
    client = FakeUSClient(fx=1380.0)
    bot = _new_bot("USD", client)
    bot.cash = 0
    bot.position = 1
    bot.qty = 1
    bot._entry_atr = 5.0

    asyncio.run(bot._execute("SELL", 140.0))

    # tax_us = 0, commission = 0.00015 → proceeds_native = 140 * 1 * 0.99985
    expected_proceeds = int(140.0 * 1 * (1 - 0.00015 - 0.0) * 1380.0)
    assert bot.cash == expected_proceeds
    assert bot.position == 0
    assert bot._highest_since_entry == 0
    assert bot._entry_atr == 0.0


def test_buy_krw_path_unchanged():
    """KRW bot should compute cash exactly as before (FX=1)."""
    client = FakeKRClient()
    bot = _new_bot("KRW", client)
    bot.cash = 1_000_000
    bot.position = 0
    bot.qty = 1

    asyncio.run(bot._execute("BUY", 70_000))

    expected_cost = int(70_000 * 1 * (1 + 0.00015))
    assert bot.cash == 1_000_000 - expected_cost
    assert bot.position == 1


def test_adapter_does_not_scale_prices():
    """The Phase-3 ×100 cents hack must be gone (Patch-1 contract)."""
    import importlib
    main = importlib.import_module("main")
    adapter = main._USClientAdapter(FakeUSClient(fx=1380.0))
    price = asyncio.run(adapter.get_current_price("NVDA"))
    # FakeUSClient returns 130.50 — adapter must pass through unchanged.
    assert math.isclose(price, 130.50, rel_tol=1e-9)


def test_adapter_exposes_fx_fields():
    import importlib
    main = importlib.import_module("main")
    adapter = main._USClientAdapter(FakeUSClient(fx=1395.5))
    assert adapter._cached_usdkrw == 1395.5
    assert adapter.default_usdkrw == 1380.0
