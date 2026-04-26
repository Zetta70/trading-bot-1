"""
Patch 4 — KISUSClient.get_order_fill real implementation.

Verifies:
  * Standard ft_ccld_amt2 path → avg_fill_price = amt / qty.
  * Fallback to tr_amt when ft_ccld_amt2 missing.
  * Fallback to ovrs_excg_unpr (unit price) when amount fields missing.
  * Partial fills (multiple rows, same odno) summed correctly.
  * Rows with foreign odno filtered out.
  * Empty / non-rt_cd-0 response → 0.0 fallback (no exception).
  * Unrecognized fields → WARNING with key dump + 0.0 fallback.
  * Network exception → 0.0 fallback.
  * TR ID switches between REAL and MOCK.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.kis_us_api import KISUSClient  # noqa: E402


def _new_client(run_mode="MOCK"):
    """Build a KISUSClient with token and HTTP layer pre-stubbed."""
    c = KISUSClient(
        api_key="k", api_secret="s", acc_no="50012345-01",
        run_mode=run_mode,
    )

    async def _fake_token():
        return "TOKEN"
    c._ensure_token = _fake_token

    c._last_request = None
    c._next_response = None

    async def _fake_request(method, url, **kwargs):
        c._last_request = {
            "method": method, "url": url, "headers": kwargs.get("headers"),
            "params": kwargs.get("params"),
        }
        return c._next_response

    c._request = _fake_request
    return c


# ── Standard amount-based path ────────────────────────────────────────

def test_amt2_path_computes_avg_fill():
    c = _new_client()
    c._next_response = {
        "rt_cd": "0",
        "output": [{
            "odno": "ORD-1", "ft_ccld_qty": "5",
            "ft_ccld_amt2": "650.00",  # 5 shares * $130
        }],
    }
    out = asyncio.run(c.get_order_fill("ORD-1"))
    assert out["filled_qty"] == 5
    assert out["avg_fill_price"] == pytest.approx(130.0)


def test_falls_back_to_tr_amt_when_amt2_missing():
    c = _new_client()
    c._next_response = {
        "rt_cd": "0",
        "output": [{
            "odno": "ORD-1", "ft_ccld_qty": "2",
            "tr_amt": "300.00",  # 2 * $150
        }],
    }
    out = asyncio.run(c.get_order_fill("ORD-1"))
    assert out["filled_qty"] == 2
    assert out["avg_fill_price"] == pytest.approx(150.0)


def test_falls_back_to_unit_price_when_amount_fields_missing():
    c = _new_client()
    c._next_response = {
        "rt_cd": "0",
        "output": [{
            "odno": "ORD-1", "ft_ccld_qty": "3",
            "ovrs_excg_unpr": "200.50",  # unit price USD
        }],
    }
    out = asyncio.run(c.get_order_fill("ORD-1"))
    assert out["filled_qty"] == 3
    assert out["avg_fill_price"] == pytest.approx(200.50)


# ── Partial fills + foreign-odno filtering ────────────────────────────

def test_partial_fills_summed_to_weighted_avg():
    c = _new_client()
    c._next_response = {
        "rt_cd": "0",
        "output": [
            {"odno": "ORD-1", "ft_ccld_qty": "2", "ft_ccld_amt2": "260.00"},  # $130
            {"odno": "ORD-1", "ft_ccld_qty": "3", "ft_ccld_amt2": "393.00"},  # $131
        ],
    }
    out = asyncio.run(c.get_order_fill("ORD-1"))
    assert out["filled_qty"] == 5
    # weighted avg = (260 + 393) / 5 = 130.6
    assert out["avg_fill_price"] == pytest.approx(130.6)


def test_foreign_odno_rows_ignored():
    c = _new_client()
    c._next_response = {
        "rt_cd": "0",
        "output": [
            {"odno": "OTHER", "ft_ccld_qty": "100", "ft_ccld_amt2": "9999"},
            {"odno": "ORD-1", "ft_ccld_qty": "1", "ft_ccld_amt2": "130.0"},
        ],
    }
    out = asyncio.run(c.get_order_fill("ORD-1"))
    assert out["filled_qty"] == 1
    assert out["avg_fill_price"] == pytest.approx(130.0)


# ── Failure modes ─────────────────────────────────────────────────────

def test_empty_output_returns_zero():
    c = _new_client()
    c._next_response = {"rt_cd": "0", "output": []}
    out = asyncio.run(c.get_order_fill("ORD-1"))
    assert out == {"avg_fill_price": 0.0, "filled_qty": 0}


def test_non_zero_rt_cd_returns_zero(caplog):
    c = _new_client()
    c._next_response = {
        "rt_cd": "1", "msg_cd": "EGW00123", "msg1": "조회 결과 없음",
    }
    with caplog.at_level(logging.WARNING):
        out = asyncio.run(c.get_order_fill("ORD-1"))
    assert out == {"avg_fill_price": 0.0, "filled_qty": 0}
    assert any("US fill query error" in r.message for r in caplog.records)


def test_network_exception_returns_zero(caplog):
    c = _new_client()
    async def _boom(*a, **kw):
        raise RuntimeError("network down")
    c._request = _boom
    with caplog.at_level(logging.WARNING):
        out = asyncio.run(c.get_order_fill("ORD-1"))
    assert out == {"avg_fill_price": 0.0, "filled_qty": 0}
    assert any("US fill query failed" in r.message for r in caplog.records)


def test_unrecognized_fields_logs_warning_with_keys(caplog):
    c = _new_client()
    c._next_response = {
        "rt_cd": "0",
        "output": [{
            "odno": "ORD-1",
            "ft_ccld_qty": "1",
            "weird_amount_field_v9": "130",  # not in our chain
        }],
    }
    with caplog.at_level(logging.WARNING):
        out = asyncio.run(c.get_order_fill("ORD-1"))
    assert out == {"avg_fill_price": 0.0, "filled_qty": 0}
    msgs = [r.message for r in caplog.records]
    assert any("no recognized qty/amount/price" in m for m in msgs)
    # The available-keys hint must be present so an operator can patch
    # the field-name chain without a code dive.
    assert any("weird_amount_field_v9" in m for m in msgs)


# ── TR-ID routing ─────────────────────────────────────────────────────

def test_tr_id_switches_for_real_vs_mock():
    real = _new_client("REAL")
    real._next_response = {"rt_cd": "0", "output": []}
    asyncio.run(real.get_order_fill("ORD-1"))
    assert real._last_request["headers"]["tr_id"] == "TTTS3035R"

    mock = _new_client("MOCK")
    mock._next_response = {"rt_cd": "0", "output": []}
    asyncio.run(mock.get_order_fill("ORD-1"))
    assert mock._last_request["headers"]["tr_id"] == "VTTS3035R"


def test_url_is_inquire_ccnl():
    c = _new_client()
    c._next_response = {"rt_cd": "0", "output": []}
    asyncio.run(c.get_order_fill("ORD-1"))
    assert (
        "/uapi/overseas-stock/v1/trading/inquire-ccnl"
        in c._last_request["url"]
    )
