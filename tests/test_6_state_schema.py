"""
Patch 6 — state.json schema versioning + migration seam.

Cases:
  * save_state writes schema_version = SCHEMA_VERSION.
  * load_state accepts a v2 file (round-trip).
  * load_state rejects a file with no schema_version (pre-v2 / hand-edited).
  * load_state rejects a file with a different version + WARNING.
  * _migrate_state seam exists, returns None (no migration wired yet),
    and is invoked when version mismatches and is not None.
  * last_sync_time round-trips through save / load (Patch 3 ↔ Patch 6).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import portfolio_manager as pm_mod  # noqa: E402
from portfolio_manager import (  # noqa: E402
    PortfolioManager, SCHEMA_VERSION,
)


class FakePredictor:
    _ml_model = None
    def score(self, _): return 0.5


class FakeKRClient:
    async def get_balance(self):
        return {"deposit": 1_000_000, "total_eval": 0, "holdings": []}

    async def get_current_price(self, _):
        return 70_000


class FakeScanner:
    async def scan(self): return []


def _new_pm():
    return PortfolioManager(
        client=FakeKRClient(),
        scanner=FakeScanner(),
        predictor=FakePredictor(),
        initial_cash=1_000_000,
        max_drawdown_pct=0.03,
        max_active_bots=5,
    )


# ── save_state writes schema_version ──────────────────────────────────

def test_save_state_writes_schema_version(tmp_path, monkeypatch):
    monkeypatch.setattr(pm_mod, "STATE_FILE", str(tmp_path / "state.json"))
    pm = _new_pm()
    pm.save_state()
    raw = json.loads(Path(pm_mod.STATE_FILE).read_text())
    assert raw["schema_version"] == SCHEMA_VERSION
    assert SCHEMA_VERSION == 2  # current code is v2 per spec


# ── Round-trip ────────────────────────────────────────────────────────

def test_v2_state_roundtrips_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(pm_mod, "STATE_FILE", str(tmp_path / "state.json"))
    pm = _new_pm()
    pm._remaining_cash = 333_333
    pm._peak_equity = 1_500_000
    pm._last_sync_time = "2025-04-26T10:00:00+09:00"
    pm.save_state()

    pm2 = _new_pm()
    assert pm2.load_state() is True
    assert pm2._remaining_cash == 333_333
    assert pm2._peak_equity == 1_500_000
    assert pm2._last_sync_time == "2025-04-26T10:00:00+09:00"
    assert pm2._state_loaded is True


# ── Rejection paths ───────────────────────────────────────────────────

def test_state_without_schema_version_rejected(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(pm_mod, "STATE_FILE", str(tmp_path / "state.json"))
    Path(pm_mod.STATE_FILE).write_text(json.dumps({
        # Note: NO schema_version key.
        "remaining_cash": 999_999,
        "peak_equity": 999_999,
        "bots": {},
    }))

    pm = _new_pm()
    with caplog.at_level(logging.WARNING):
        ok = pm.load_state()

    assert ok is False
    assert pm._state_loaded is False
    assert any(
        "no schema_version" in r.message
        and "starting fresh" in r.message.lower()
        for r in caplog.records
    )
    # Critical: defaults must NOT be silently applied from the old file.
    assert pm._remaining_cash == 1_000_000  # untouched
    assert pm._peak_equity == 1_000_000


def test_state_with_wrong_version_rejected_via_migration(
    tmp_path, monkeypatch, caplog,
):
    """A v1 file should hit _migrate_state, which currently returns None
    (no migration wired) and load_state should refuse."""
    monkeypatch.setattr(pm_mod, "STATE_FILE", str(tmp_path / "state.json"))
    Path(pm_mod.STATE_FILE).write_text(json.dumps({
        "schema_version": 1,
        "remaining_cash": 777_777,
        "peak_equity": 777_777,
        "bots": {},
    }))

    pm = _new_pm()
    with caplog.at_level(logging.WARNING):
        ok = pm.load_state()

    assert ok is False
    assert pm._remaining_cash == 1_000_000  # untouched
    assert any(
        "No migration implemented from schema v1" in r.message
        for r in caplog.records
    )


# ── Migration seam ────────────────────────────────────────────────────

def test_migrate_state_seam_exists_and_returns_none():
    out = PortfolioManager._migrate_state({"k": "v"}, 1, 2)
    assert out is None  # seam present, no migration wired yet


def test_migrate_state_can_be_overridden_by_subclass(tmp_path, monkeypatch):
    """Future migrations: subclass can override _migrate_state to map
    old schemas forward. Verify the override path is honored."""
    monkeypatch.setattr(pm_mod, "STATE_FILE", str(tmp_path / "state.json"))
    Path(pm_mod.STATE_FILE).write_text(json.dumps({
        "schema_version": 1,
        "remaining_cash": 555_555,
        "peak_equity": 600_000,
        "bots": {},
    }))

    class MigratingPM(PortfolioManager):
        @staticmethod
        def _migrate_state(old, from_v, to_v):
            # Trivial migration: just re-stamp the version.
            old["schema_version"] = to_v
            return old

    pm = MigratingPM(
        client=FakeKRClient(),
        scanner=FakeScanner(),
        predictor=FakePredictor(),
        initial_cash=1_000_000,
    )
    assert pm.load_state() is True
    assert pm._remaining_cash == 555_555
    assert pm._peak_equity == 600_000


# ── Cross-patch interaction ───────────────────────────────────────────

def test_last_sync_time_round_trips_alongside_schema(tmp_path, monkeypatch):
    """Patch 3's last_sync_time + Patch 6's schema_version must coexist."""
    monkeypatch.setattr(pm_mod, "STATE_FILE", str(tmp_path / "state.json"))
    pm = _new_pm()
    asyncio.run(pm.sync_balance())
    iso = pm._last_sync_time
    assert iso is not None
    pm.save_state()

    pm2 = _new_pm()
    assert pm2.load_state() is True
    assert pm2._last_sync_time == iso

    raw = json.loads(Path(pm_mod.STATE_FILE).read_text())
    assert "schema_version" in raw
    assert "last_sync_time" in raw
