"""
Patch 2 — dynamic qty in TradingBot._execute.

Sizing rules:
  BUY:  refuse if a position already exists (no averaging up);
        size to floor((cash * 0.95) / cost_per_share_krw).
  SELL: full liquidation; refuse if no position.

Also verifies:
  * `qty=1` hard-code is gone from main.py bot_kwargs.
  * Constructing TradingBot(qty=...) emits a DeprecationWarning.
  * The trade-log row records the actual qty traded (not self.qty).
"""

from __future__ import annotations

import asyncio
import re
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_bot import TradingBot  # noqa: E402


# ── Fakes ─────────────────────────────────────────────────────────────

class FakePredictor:
    def score(self, _):
        return 0.5


class FakeUSClient:
    def __init__(self, fx=1380.0):
        self._cached_usdkrw = fx
        self.default_usdkrw = 1380.0
        self.last_order = None

    async def place_order(self, ticker, qty, side, order_type="market", price=0):
        self.last_order = {
            "ticker": ticker, "qty": qty, "side": side,
            "order_type": order_type, "price": price,
        }
        return {"output": {"ODNO": "ORD-2"}}

    async def get_order_fill(self, _):
        return {"avg_fill_price": 0, "filled_qty": 0}


class FakeKRClient:
    def __init__(self):
        self.last_order = None

    async def place_order(self, ticker, qty, side, order_type="market", price=0):
        self.last_order = {
            "ticker": ticker, "qty": qty, "side": side,
            "order_type": order_type, "price": price,
        }
        return {"output": {"ODNO": "ORD-2-KR"}}

    async def get_order_fill(self, _):
        return {"avg_fill_price": 0, "filled_qty": 0}


def _new_bot(currency, client, cash):
    bot = TradingBot(
        client=client,
        ticker="NVDA" if currency == "USD" else "005930",
        predictor=FakePredictor(),
        currency=currency,
        initial_cash=cash,
        commission=0.00015,
        tax_kr=0.0023,
        tax_us=0.0,
    )
    bot.cash = cash
    bot.position = 0
    return bot


# ── Sizing math ───────────────────────────────────────────────────────

def test_buy_sizes_to_one_share_when_cash_is_tight():
    """200k KRW / ($130 * 1.001 * 1380) ≈ 1.06 → 1 share."""
    client = FakeUSClient(fx=1380.0)
    bot = _new_bot("USD", client, cash=200_000)

    asyncio.run(bot._execute("BUY", 130.0))

    assert bot.position == 1
    assert client.last_order["qty"] == 1


def test_buy_sizes_to_five_shares_with_more_cash():
    """1_000_000 KRW / (130 * 1.00115 * 1380) ≈ 5.3 → 5 shares."""
    client = FakeUSClient(fx=1380.0)
    bot = _new_bot("USD", client, cash=1_000_000)

    asyncio.run(bot._execute("BUY", 130.0))

    assert bot.position == 5
    assert client.last_order["qty"] == 5


def test_buy_skipped_when_already_holding():
    """Existing position must NOT be added to (no averaging up)."""
    client = FakeUSClient(fx=1380.0)
    bot = _new_bot("USD", client, cash=1_000_000)
    bot.position = 3
    bot.last_price = 130.0

    asyncio.run(bot._execute("BUY", 130.0))

    assert bot.position == 3  # unchanged
    assert client.last_order is None  # no order placed


def test_buy_skipped_when_cash_insufficient_for_one_share():
    client = FakeUSClient(fx=1380.0)
    bot = _new_bot("USD", client, cash=10_000)  # ~$7 — not enough

    asyncio.run(bot._execute("BUY", 130.0))

    assert bot.position == 0
    assert client.last_order is None


def test_buy_uses_95pct_buffer():
    """Cash 200k, price 80k KRW → usable=190k → 190/80.092 ≈ 2.37 → 2 shares.
    With no buffer it would be 200/80 = 2.5 → 2 anyway, so use a tighter
    edge case: cash 100k, price 50k → usable 95k → 1.899 → 1 share
    (no-buffer would give 100/50 = 2)."""
    client = FakeKRClient()
    bot = _new_bot("KRW", client, cash=100_000)

    asyncio.run(bot._execute("BUY", 50_000))

    assert bot.position == 1, "5% cash buffer must reduce qty from 2 to 1"


# ── SELL behavior ─────────────────────────────────────────────────────

def test_sell_liquidates_full_position():
    client = FakeUSClient(fx=1380.0)
    bot = _new_bot("USD", client, cash=0)
    bot.position = 5
    bot._highest_since_entry = 145.0
    bot._entry_atr = 6.0

    asyncio.run(bot._execute("SELL", 140.0))

    assert bot.position == 0
    assert client.last_order["qty"] == 5
    assert bot._highest_since_entry == 0
    assert bot._entry_atr == 0.0


def test_sell_skipped_when_no_position():
    client = FakeUSClient(fx=1380.0)
    bot = _new_bot("USD", client, cash=0)
    bot.position = 0

    asyncio.run(bot._execute("SELL", 140.0))

    assert client.last_order is None


# ── Deprecation + main.py prune ───────────────────────────────────────

def test_qty_kwarg_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TradingBot(
            client=FakeUSClient(),
            ticker="NVDA",
            predictor=FakePredictor(),
            currency="USD",
            qty=3,
            initial_cash=100_000,
        )
    assert any(
        issubclass(rec.category, DeprecationWarning)
        and "qty" in str(rec.message)
        for rec in w
    ), "TradingBot(qty=...) should warn"


def test_qty_unset_does_not_warn():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TradingBot(
            client=FakeUSClient(),
            ticker="NVDA",
            predictor=FakePredictor(),
            currency="USD",
            initial_cash=100_000,
        )
    qty_warnings = [
        rec for rec in w
        if issubclass(rec.category, DeprecationWarning)
        and "qty" in str(rec.message)
    ]
    assert qty_warnings == []


def test_main_py_no_longer_hardcodes_qty_1():
    """Source-level guard: neither bot_kwargs block hard-codes qty=1."""
    main_src = (Path(__file__).resolve().parent.parent / "main.py").read_text()
    # Match `"qty": 1` anywhere outside comments. A simple regex over
    # the file is enough — the hard-code was the only occurrence.
    matches = re.findall(r'^\s*"qty":\s*1\b', main_src, flags=re.MULTILINE)
    assert matches == [], (
        "main.py still hard-codes 'qty': 1 in bot_kwargs — Patch 2 "
        "should have removed it."
    )


def test_trade_log_records_actual_qty(tmp_path, monkeypatch):
    """The CSV row's qty column reflects the dynamically-sized qty."""
    import trading_bot
    log_path = tmp_path / "trade_log.csv"
    monkeypatch.setattr(trading_bot, "TRADE_LOG", str(log_path))
    # Re-init the CSV with header.
    trading_bot._init_csv(str(log_path), trading_bot.TRADE_HEADER)

    client = FakeUSClient(fx=1380.0)
    bot = _new_bot("USD", client, cash=1_000_000)
    asyncio.run(bot._execute("BUY", 130.0))

    rows = log_path.read_text().splitlines()
    assert len(rows) >= 2  # header + 1 trade
    # Header: timestamp,ticker,side,price,fill_price,qty,order_no,equity,ml_score
    last = rows[-1].split(",")
    qty_col = int(last[5])
    assert qty_col == bot.position == 5
