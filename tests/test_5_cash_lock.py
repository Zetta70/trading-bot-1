"""
Patch 5 — asyncio.Lock guards cash mutations against concurrent races.

Cases:
  * PortfolioManager creates a single asyncio.Lock and injects it into
    bot_kwargs as `cash_lock`.
  * Concurrent add_bot calls never push _remaining_cash negative.
  * Concurrent _execute calls never tear cash arithmetic; final cash =
    initial - sum(costs) regardless of interleaving.
  * Lock is NOT held across the network await — bots can place orders
    in parallel.
  * TradingBot without a lock (legacy / tests) still works via the
    no-op context manager.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_manager import PortfolioManager  # noqa: E402
from trading_bot import TradingBot, _NullAsyncLock  # noqa: E402


# ── Fakes ─────────────────────────────────────────────────────────────

class FakePredictor:
    _ml_model = None
    def score(self, _): return 0.5


class SlowKRClient:
    """A client whose place_order awaits asyncio.sleep so we can
    interleave multiple bots' execute() calls under the scheduler."""

    def __init__(self):
        self.concurrent_in_flight = 0
        self.peak_in_flight = 0
        self._lock = asyncio.Lock()

    async def get_balance(self):
        return {"deposit": 1_000_000, "total_eval": 0, "holdings": []}

    async def get_current_price(self, _):
        return 70_000

    async def place_order(self, ticker, qty, side, order_type="market", price=0):
        async with self._lock:
            self.concurrent_in_flight += 1
            self.peak_in_flight = max(
                self.peak_in_flight, self.concurrent_in_flight,
            )
        # Yield to the event loop so other coroutines can run.
        await asyncio.sleep(0.05)
        async with self._lock:
            self.concurrent_in_flight -= 1
        return {"output": {"ODNO": f"ORD-{ticker}-{side}"}}

    async def get_order_fill(self, _):
        return {"avg_fill_price": 0, "filled_qty": 0}


class FakeScanner:
    async def scan(self): return []


def _new_pm(client):
    return PortfolioManager(
        client=client,
        scanner=FakeScanner(),
        predictor=FakePredictor(),
        initial_cash=1_000_000,
        max_drawdown_pct=0.03,
        max_active_bots=10,
    )


# ── Wiring ────────────────────────────────────────────────────────────

def test_portfolio_manager_creates_lock_and_injects_into_bot_kwargs():
    pm = _new_pm(SlowKRClient())
    assert isinstance(pm._cash_lock, asyncio.Lock)
    assert pm._bot_kwargs["cash_lock"] is pm._cash_lock


# ── Concurrent add_bot ────────────────────────────────────────────────

def test_concurrent_add_bot_never_overcommits_cash():
    """5 add_bot calls firing simultaneously must not drive
    _remaining_cash below zero."""
    pm = _new_pm(SlowKRClient())
    pm.max_active_bots = 5
    # Force allocation to take >100% of cash if races aren't guarded.
    # _cash_per_bot() = 1_000_000 // 5 = 200_000 each. With 6 concurrent
    # adds and only 5 max bots, we test both the cash guard and the
    # max-bots guard interacting with the lock.

    async def go():
        coros = [pm.add_bot(f"00000{i}") for i in range(1, 7)]
        return await asyncio.gather(*coros)

    asyncio.run(go())
    assert pm._remaining_cash >= 0
    # Total allocated == initial - remaining; should equal 5 * 200_000.
    allocated = 1_000_000 - pm._remaining_cash
    assert allocated == 5 * 200_000


# ── Concurrent _execute ───────────────────────────────────────────────

def test_concurrent_execute_does_not_tear_cash():
    """Multiple bots' _execute calls should each correctly mutate
    their own cash atomically — final = initial - cost per bot.
    Also: the network call (place_order) runs in parallel — the lock
    must NOT serialize bots."""
    client = SlowKRClient()
    pm = _new_pm(client)

    bots = []
    for i in range(5):
        bot = TradingBot(
            client=client,
            ticker=f"00000{i+1}",
            predictor=FakePredictor(),
            currency="KRW",
            initial_cash=200_000,
            cash_lock=pm._cash_lock,
        )
        bots.append(bot)

    async def buy_all():
        await asyncio.gather(*(b._execute("BUY", 50_000) for b in bots))

    asyncio.run(buy_all())

    # Each bot bought 1 share at 50k (200k * 0.95 / (50k * 1.001) ≈ 3.79
    # → 3 shares actually). Verify cash arithmetic is sane regardless.
    for bot in bots:
        assert bot.position >= 1
        assert bot.cash >= 0
        # Cash must be exactly initial - cost (no torn writes).
        expected_cost = int(50_000 * bot.position * (1 + 0.00015))
        assert bot.cash == 200_000 - expected_cost

    # Critical: the network calls ran in parallel — peak in-flight
    # should be > 1, otherwise the lock is wrapping the await and
    # serializing all bots.
    assert client.peak_in_flight >= 2, (
        "place_order calls were serialized — the cash lock is "
        "wrapping the network await (anti-pattern)."
    )


# ── Lock-less mode ────────────────────────────────────────────────────

def test_bot_without_lock_uses_null_context():
    bot = TradingBot(
        client=SlowKRClient(),
        ticker="005930",
        predictor=FakePredictor(),
        currency="KRW",
        initial_cash=100_000,
    )
    ctx = bot._cash_lock_ctx()
    assert isinstance(ctx, _NullAsyncLock)


def test_null_lock_is_an_async_context_manager():
    async def go():
        async with _NullAsyncLock():
            return "ok"

    assert asyncio.run(go()) == "ok"
