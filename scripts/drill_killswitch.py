"""
Kill-switch Drill — rehearse catastrophe scenarios in paper mode.

Scenarios:
  1. Network outage during open position
  2. API returns malformed data / 5xx
  3. Unexpected portfolio drawdown breach
  4. Simultaneous stop-outs across all positions

Run in MOCK mode only. Refuses to run in REAL mode.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root on sys.path when invoked directly.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


async def drill_scenario_drawdown(portfolio) -> None:
    """
    Inject a synthetic -10% portfolio drawdown by monkey-patching
    ``total_equity`` and verify ``_check_drawdown`` raises KillSwitchError.
    """
    from portfolio_manager import KillSwitchError

    logger.info("DRILL: Injecting synthetic -10% portfolio drawdown")
    target_equity = portfolio.total_equity * 0.89
    cls = type(portfolio)
    original_prop = cls.total_equity
    cls.total_equity = property(lambda self: target_equity)
    try:
        portfolio._check_drawdown()
        logger.error("DRILL FAIL: kill-switch did NOT trigger")
    except KillSwitchError as e:
        logger.info("DRILL PASS: kill-switch triggered as expected: %s", e)
    except Exception as e:
        logger.warning(
            "DRILL UNEXPECTED: raised %s instead of KillSwitchError: %s",
            type(e).__name__, e,
        )
    finally:
        cls.total_equity = original_prop


async def drill_scenario_api_failure(client) -> None:
    """Simulate KIS API 5xx and verify the exception propagates."""
    logger.info("DRILL: Simulating KIS API 500 error")
    original = client._request

    async def broken(*args, **kwargs):
        raise RuntimeError("Simulated 500")

    client._request = broken
    try:
        try:
            await client.get_current_price("005930")
            logger.error("DRILL FAIL: expected exception did not propagate")
        except Exception as e:
            logger.info("DRILL PASS: exception handled: %s", e)
    finally:
        client._request = original


async def main() -> None:
    from config import Config
    from kis_client import KISClient

    cfg = Config()
    if cfg.run_mode != "MOCK":
        logger.error("DRILL MUST RUN IN MOCK MODE ONLY. Exiting.")
        sys.exit(1)

    client = KISClient(
        cfg.api_key, cfg.api_secret, cfg.acc_no, run_mode="MOCK",
    )
    try:
        await drill_scenario_api_failure(client)
    finally:
        await client.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    asyncio.run(main())
