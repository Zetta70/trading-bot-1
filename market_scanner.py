"""
Market Scanner — discovers high-potential tickers dynamically.

In MOCK mode, simulates volume/price surge discovery from a known universe.
In REAL mode, queries KIS volume ranking API (placeholder for implementation).

Runs every scan_interval seconds (default 15 min) and returns ranked candidates.
"""

import random
import logging

logger = logging.getLogger(__name__)

# Full universe of stocks for mock scanning
UNIVERSE = [
    ("005930", "Samsung Electronics"),
    ("000660", "SK Hynix"),
    ("035420", "NAVER"),
    ("035720", "Kakao"),
    ("051910", "LG Chem"),
    ("006400", "Samsung SDI"),
    ("068270", "Celltrion"),
    ("005380", "Hyundai Motor"),
    ("000270", "Kia"),
    ("105560", "KB Financial"),
    ("055550", "Shinhan Financial"),
    ("096770", "SK Innovation"),
    ("003670", "POSCO Holdings"),
    ("028260", "Samsung C&T"),
    ("207940", "Samsung Biologics"),
]


class MarketScanner:
    """
    Scans the market for volume/momentum breakout candidates.

    Returns a list of dicts sorted by potential:
        {"ticker": str, "name": str, "price": int,
         "volume_rank": int, "price_change_pct": float}
    """

    def __init__(self, client, mock_mode: bool = True):
        self.client = client
        self.mock_mode = mock_mode

    async def scan(self) -> list[dict]:
        """Run a scan and return ranked candidate tickers."""
        if self.mock_mode:
            return await self._mock_scan()
        else:
            return await self._real_scan()

    async def _mock_scan(self) -> list[dict]:
        """Simulate discovering 3–5 tickers with volume surges."""
        count = random.randint(3, 5)
        picks = random.sample(UNIVERSE, count)

        results = []
        for ticker, name in picks:
            price = await self.client.get_current_price(ticker)
            results.append({
                "ticker": ticker,
                "name": name,
                "price": price,
                "volume_rank": random.randint(1, 50),
                "price_change_pct": random.uniform(-1.5, 3.0),
            })

        results.sort(key=lambda x: x["volume_rank"])

        logger.info(
            "Scanner found %d candidates: %s",
            len(results),
            [f"{r['ticker']}({r['name']})" for r in results],
        )
        return results

    async def _real_scan(self) -> list[dict]:
        """
        Query KIS volume ranking API.

        Placeholder — when implementing for real, use:
          GET /uapi/domestic-stock/v1/ranking/volume
          tr_id = FHPST01710000
        """
        logger.warning(
            "Real market scan not yet implemented. "
            "Returning empty list — seed tickers will be used."
        )
        return []
