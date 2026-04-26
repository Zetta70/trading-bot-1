"""
Patch 3 — PortfolioManager.sync_balance() reconciles broker deposit
with .env INITIAL_CASH and persists last_sync_time.

Cases:
  * Broker deposit < env limit → uses broker amount.
  * Broker deposit > env limit → caps at env amount.
  * Broker raises → falls back to env, no exception, WARNING logged.
  * Broker returns 0 deposit → falls back to env, WARNING logged.
  * No state loaded → _remaining_cash gets the synced value.
  * State loaded → _remaining_cash preserved (already-allocated capital).
  * Sync timestamp <24h → API NOT called (TTL skip).
  * Sync timestamp >24h → API IS called.
  * _USClientAdapter.get_balance() converts USD→KRW via FX.
  * last_sync_time persists round-trip via save_state/load_state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import KST  # noqa: E402
from portfolio_manager import PortfolioManager  # noqa: E402


# ── Fakes ─────────────────────────────────────────────────────────────

class FakePredictor:
    _ml_model = None
    def score(self, _): return 0.5


class FakeKRClient:
    def __init__(self, deposit_krw=850_000, raise_on_balance=False):
        self._deposit = deposit_krw
        self._raise = raise_on_balance
        self.balance_calls = 0

    async def get_balance(self):
        self.balance_calls += 1
        if self._raise:
            raise RuntimeError("network error")
        return {"deposit": self._deposit, "total_eval": 0, "holdings": []}

    async def get_current_price(self, _):
        return 70_000


class FakeUSClient:
    def __init__(self, deposit_usd=600.0, fx=1380.0):
        self._cached_usdkrw = fx
        self.default_usdkrw = 1380.0
        self._deposit_usd = deposit_usd
        self.balance_calls = 0

    async def get_us_balance(self):
        self.balance_calls += 1
        return {
            "deposit_usd": self._deposit_usd,
            "total_eval_usd": 0.0,
            "total_pnl_usd": 0.0,
            "holdings": [],
        }


class FakeScanner:
    async def scan(self):
        return []


def _new_pm(client, env_initial_cash=1_000_000):
    return PortfolioManager(
        client=client,
        scanner=FakeScanner(),
        predictor=FakePredictor(),
        initial_cash=env_initial_cash,
        max_drawdown_pct=0.03,
        max_active_bots=5,
    )


# ── Sync math ─────────────────────────────────────────────────────────

def test_sync_uses_broker_when_below_env_limit():
    client = FakeKRClient(deposit_krw=850_000)
    pm = _new_pm(client, env_initial_cash=1_000_000)

    asyncio.run(pm.sync_balance())

    assert pm.initial_cash == 850_000
    assert pm._remaining_cash == 850_000
    assert pm._last_sync_time is not None


def test_sync_caps_at_env_when_broker_higher():
    client = FakeKRClient(deposit_krw=2_000_000)
    pm = _new_pm(client, env_initial_cash=1_000_000)

    asyncio.run(pm.sync_balance())

    assert pm.initial_cash == 1_000_000  # capped
    assert pm._remaining_cash == 1_000_000


def test_sync_falls_back_on_broker_error(caplog):
    client = FakeKRClient(raise_on_balance=True)
    pm = _new_pm(client, env_initial_cash=1_000_000)

    with caplog.at_level(logging.WARNING):
        asyncio.run(pm.sync_balance())

    assert pm.initial_cash == 1_000_000  # unchanged
    assert pm._last_sync_time is None
    assert any("Account sync failed" in r.message for r in caplog.records)


def test_sync_falls_back_on_zero_deposit(caplog):
    client = FakeKRClient(deposit_krw=0)
    pm = _new_pm(client, env_initial_cash=1_000_000)

    with caplog.at_level(logging.WARNING):
        asyncio.run(pm.sync_balance())

    assert pm.initial_cash == 1_000_000  # unchanged
    assert pm._last_sync_time is None
    assert any("zero or missing" in r.message for r in caplog.records)


def test_sync_preserves_remaining_cash_when_state_loaded():
    """If state was loaded, _remaining_cash already reflects what's
    free vs allocated to bots — sync must not clobber that."""
    client = FakeKRClient(deposit_krw=900_000)
    pm = _new_pm(client, env_initial_cash=1_000_000)
    pm._state_loaded = True
    pm._remaining_cash = 250_000  # representing pre-existing allocation

    asyncio.run(pm.sync_balance())

    assert pm.initial_cash == 900_000
    assert pm._remaining_cash == 250_000  # untouched


# ── 24h TTL ───────────────────────────────────────────────────────────

def test_sync_ttl_skips_recent_sync():
    client = FakeKRClient(deposit_krw=900_000)
    pm = _new_pm(client, env_initial_cash=1_000_000)
    # Pretend a sync happened 1 hour ago.
    pm._last_sync_time = (
        datetime.now(KST) - timedelta(hours=1)
    ).isoformat()

    asyncio.run(pm.sync_balance())

    assert client.balance_calls == 0  # API NOT called
    assert pm.initial_cash == 1_000_000  # unchanged


def test_sync_ttl_resyncs_after_24h():
    client = FakeKRClient(deposit_krw=900_000)
    pm = _new_pm(client, env_initial_cash=1_000_000)
    pm._last_sync_time = (
        datetime.now(KST) - timedelta(hours=25)
    ).isoformat()

    asyncio.run(pm.sync_balance())

    assert client.balance_calls == 1
    assert pm.initial_cash == 900_000


def test_sync_ttl_corrupted_timestamp_falls_through(caplog):
    client = FakeKRClient(deposit_krw=900_000)
    pm = _new_pm(client, env_initial_cash=1_000_000)
    pm._last_sync_time = "not-an-iso-date"

    asyncio.run(pm.sync_balance())

    assert client.balance_calls == 1


# ── US adapter ────────────────────────────────────────────────────────

def test_adapter_get_balance_converts_usd_to_krw():
    """The _USClientAdapter must return ``deposit`` in KRW so
    PortfolioManager can sync market-agnostically."""
    import importlib
    main = importlib.import_module("main")
    us = FakeUSClient(deposit_usd=600.0, fx=1380.0)
    adapter = main._USClientAdapter(us)

    bal = asyncio.run(adapter.get_balance())

    assert bal["deposit_usd"] == 600.0
    assert bal["deposit"] == int(600.0 * 1380.0)
    assert bal["fx_rate"] == 1380.0


def test_us_session_sync_path_end_to_end():
    """PortfolioManager pointed at the adapter should sync USD→KRW."""
    import importlib
    main = importlib.import_module("main")
    us = FakeUSClient(deposit_usd=500.0, fx=1380.0)
    adapter = main._USClientAdapter(us)
    pm = _new_pm(adapter, env_initial_cash=1_000_000)

    asyncio.run(pm.sync_balance())

    expected_krw = int(500.0 * 1380.0)  # 690,000
    assert pm.initial_cash == min(expected_krw, 1_000_000)
    assert pm._remaining_cash == pm.initial_cash


# ── State round-trip ──────────────────────────────────────────────────

def test_last_sync_time_persists_through_save_load(tmp_path, monkeypatch):
    import portfolio_manager as pm_mod
    monkeypatch.setattr(pm_mod, "STATE_FILE", str(tmp_path / "state.json"))

    client = FakeKRClient(deposit_krw=900_000)
    pm = _new_pm(client, env_initial_cash=1_000_000)
    asyncio.run(pm.sync_balance())
    saved_iso = pm._last_sync_time
    pm.save_state()

    raw = json.loads(Path(pm_mod.STATE_FILE).read_text())
    assert raw["last_sync_time"] == saved_iso

    pm2 = _new_pm(FakeKRClient(deposit_krw=999_999), env_initial_cash=1_000_000)
    assert pm2.load_state() is True
    assert pm2._last_sync_time == saved_iso
    assert pm2._state_loaded is True
