"""
Market Scanner — discovers high-potential tickers dynamically.

In MOCK mode, simulates volume/price surge discovery from a known universe.
In REAL mode, queries KIS volume ranking API (placeholder for implementation).

Runs every scan_interval seconds (default 15 min) and returns ranked candidates.
"""

import random
import logging

import numpy as np
import pandas as pd

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
        Query KIS volume ranking API for real-mode scanning.

        Endpoint: /uapi/domestic-stock/v1/quotations/volume-rank
        TR ID: FHPST01710000
        """
        if not hasattr(self.client, "_ensure_token"):
            logger.error(
                "Real scan requires KISClient with token support. "
                "Scanner returning empty list — only seed tickers will trade."
            )
            return []

        try:
            token = await self.client._ensure_token()
            url = (
                f"{self.client.base_url}"
                "/uapi/domestic-stock/v1/quotations/volume-rank"
            )
            headers = self.client._headers("FHPST01710000", token)
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_COND_SCR_DIV_CODE": "20171",
                "FID_INPUT_ISCD": "0000",
                "FID_DIV_CLS_CODE": "0",
                "FID_BLNG_CLS_CODE": "0",
                "FID_TRGT_CLS_CODE": "111111111",
                "FID_TRGT_EXLS_CLS_CODE": "0000000000",
                "FID_INPUT_PRICE_1": "0",
                "FID_INPUT_PRICE_2": "0",
                "FID_VOL_CNT": "0",
                "FID_INPUT_DATE_1": "0",
            }
            data = await self.client._request(
                "GET", url, headers=headers, params=params,
            )

            if data.get("rt_cd") != "0":
                logger.warning(
                    "Volume rank API error: [%s] %s",
                    data.get("msg_cd"), data.get("msg1"),
                )
                return []

            results = []
            for row in data.get("output", [])[:20]:
                try:
                    results.append({
                        "ticker": row.get("mksc_shrn_iscd", ""),
                        "name": row.get("hts_kor_isnm", ""),
                        "price": int(row.get("stck_prpr", "0") or 0),
                        "volume_rank": int(row.get("data_rank", "0") or 0),
                        "price_change_pct": float(
                            row.get("prdy_ctrt", "0") or 0,
                        ),
                    })
                except (ValueError, TypeError) as e:
                    logger.debug("Skipping malformed row: %s", e)
                    continue

            logger.info("Real scanner found %d candidates", len(results))
            return results

        except Exception as e:
            logger.error("Real scan failed: %s", e)
            return []


class CrossSectionalScanner:
    """
    Daily cross-sectional ranking scanner.

    Each day every stock in the universe is scored on a multi-factor
    composite; only the top-N are returned.

    Factors
    -------
    momentum_126d : 6-month return, skipping the most recent 20 trading
                    days to avoid the short-term mean-reversion effect.
    quality       : inverse 60-day annualized volatility.
    ml_edge       : predicted probability - 0.5 (only if an ML model is
                    loaded on the predictor).
    """

    def __init__(
        self,
        predictor,
        universe_loader,  # callable: () -> list[ticker]
        top_n: int = 15,
        momentum_weight: float = 0.35,
        quality_weight: float = 0.20,
        ml_weight: float = 0.45,
    ):
        self.predictor = predictor
        self.universe_loader = universe_loader
        self.top_n = top_n
        self.w_mom = momentum_weight
        self.w_qual = quality_weight
        self.w_ml = ml_weight
        from ohlcv_cache import get_cache
        self._cache = get_cache()

    async def scan(self) -> list[dict]:
        tickers = self.universe_loader()
        scored = []

        for ticker in tickers:
            ohlcv = self._cache.get(ticker)
            if ohlcv is None or len(ohlcv) < 150:
                continue

            try:
                close = ohlcv["close"]
                if len(close) < 146:
                    continue

                mom = close.iloc[-20] / close.iloc[-146] - 1

                log_ret = np.log(close / close.shift(1))
                vol_60 = log_ret.tail(60).std() * np.sqrt(252)
                qual = 1.0 / (vol_60 + 0.05)

                ml_edge = 0.0
                if self.predictor._ml_model is not None:
                    try:
                        from features import add_features
                        feat = add_features(ohlcv)
                        model_cols = self.predictor._ml_model.feature_names
                        available = [
                            c for c in model_cols if c in feat.columns
                        ] if model_cols else list(feat.columns)
                        latest = feat[available].iloc[[-1]]
                        if not latest.isna().any(axis=1).iloc[0]:
                            prob = self.predictor._ml_model.predict_proba(
                                latest,
                            )[0]
                            ml_edge = float(prob) - 0.5
                    except Exception:
                        pass

                scored.append({
                    "ticker": ticker,
                    "momentum": float(mom),
                    "quality": float(qual),
                    "ml_edge": ml_edge,
                    "price": int(close.iloc[-1]),
                })
            except Exception as e:
                logger.debug("Scoring failed for %s: %s", ticker, e)
                continue

        if not scored:
            logger.warning("No scorable tickers found")
            return []

        df = pd.DataFrame(scored)
        df["mom_rank"] = df["momentum"].rank(pct=True)
        df["qual_rank"] = df["quality"].rank(pct=True)
        df["ml_rank"] = df["ml_edge"].rank(pct=True)
        df["composite"] = (
            self.w_mom * df["mom_rank"]
            + self.w_qual * df["qual_rank"]
            + self.w_ml * df["ml_rank"]
        )

        top = df.nlargest(self.top_n, "composite")
        results = []
        for _, row in top.iterrows():
            results.append({
                "ticker": row["ticker"],
                "name": "",
                "price": int(row["price"]),
                "composite_score": round(float(row["composite"]), 4),
                "momentum": round(float(row["momentum"]), 4),
                "ml_edge": round(float(row["ml_edge"]), 4),
            })

        logger.info(
            "Cross-sectional scan: top %d picks (composite range %.3f~%.3f)",
            len(results),
            float(top["composite"].min()),
            float(top["composite"].max()),
        )
        return results
