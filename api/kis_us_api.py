"""
KIS Overseas (US) Stock API Client.

Implements the KIS 해외주식 주문 V2 API for US market trading:
  - get_us_price()    : 해외주식 현재가 조회
  - place_us_order()  : 해외주식 매수/매도 주문
  - get_us_balance()  : 해외주식 잔고 조회
  - get_exchange_rate(): 실시간 환율 조회

Supports both 실전(live) and 모의투자(paper) servers.

API Reference:
  https://apiportal.koreainvestment.com/apiservice
  해외주식주문[v2] → JTTT1002U (매수), JTTT1006U (매도)
  해외주식현재가 → HHDFS00000300
  해외주식잔고  → JTTT3012R

Server URLs:
  REAL : https://openapi.koreainvestment.com:9443
  MOCK : https://openapivts.koreainvestment.com:29443
"""

from __future__ import annotations

import asyncio
import random
import time
import logging
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)

# ── Server Endpoints ─────────────────────────────────────────────────
_SERVERS = {
    "REAL": "https://openapi.koreainvestment.com:9443",
    "MOCK": "https://openapivts.koreainvestment.com:29443",
}

# ── Transaction IDs for overseas stocks ──────────────────────────────
_US_TR_IDS = {
    "REAL": {
        "price":   "HHDFS00000300",    # 해외주식 현재가
        "buy":     "JTTT1002U",        # 해외주식 매수 V2
        "sell":    "JTTT1006U",        # 해외주식 매도 V2
        "balance": "JTTT3012R",        # 해외주식 잔고
    },
    "MOCK": {
        "price":   "HHDFS00000300",
        "buy":     "VTTT1002U",        # 모의 매수
        "sell":    "VTTT1006U",        # 모의 매도
        "balance": "VTTT3012R",        # 모의 잔고
    },
}

# ── Exchange code mapping ────────────────────────────────────────────
# KIS API requires exchange code for overseas stocks
_EXCHANGE_CODES = {
    "NYSE":   "NYS",
    "NASDAQ": "NAS",
    "AMEX":   "AMS",
}

# Default to NASDAQ for most US tech stocks; caller can override
_DEFAULT_EXCHANGE = "NAS"


@dataclass
class _TokenInfo:
    """OAuth2 token with expiry."""
    access_token: str
    expires_at: float


class KISUSClient:
    """
    Async REST client for KIS overseas (US) stock trading.

    Mirrors the KISClient interface but routes to 해외주식 API endpoints.
    Shares the same OAuth2 token format and retry logic.

    Parameters
    ----------
    api_key : KIS_US_API_KEY (or shared KIS_API_KEY)
    api_secret : KIS_US_API_SECRET (or shared KIS_API_SECRET)
    acc_no : Account number in "XXXXXXXX-XX" format
    run_mode : "REAL" or "MOCK"
    default_usdkrw : Fallback USD/KRW rate
    """

    TOKEN_REFRESH_MARGIN = 60
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 15  # Overseas API can be slower

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        acc_no: str,
        run_mode: str = "MOCK",
        default_usdkrw: float = 1380.0,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.run_mode = run_mode.upper()
        self.default_usdkrw = default_usdkrw

        parts = acc_no.split("-")
        self.cano = parts[0]
        self.acnt_prdt_cd = parts[1] if len(parts) > 1 else "01"

        self.base_url = _SERVERS.get(self.run_mode, _SERVERS["MOCK"])
        self._tr = _US_TR_IDS.get(self.run_mode, _US_TR_IDS["MOCK"])

        self._token: _TokenInfo | None = None
        self._session: aiohttp.ClientSession | None = None
        self._token_lock = asyncio.Lock()

        # Cached exchange rate
        self._cached_usdkrw: float = default_usdkrw
        self._fx_cache_time: float = 0

        mode_label = "실전투자" if self.run_mode == "REAL" else "모의투자"
        logger.info(
            "KIS US Client [%s] → %s (계좌: %s-%s)",
            mode_label, self.base_url, self.cano, self.acnt_prdt_cd,
        )

    # ═════════════════════════════════════════════════════════════════
    # Session & Retry
    # ═════════════════════════════════════════════════════════════════

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT),
            )
        return self._session

    async def _request(
        self,
        method: str,
        url: str,
        *,
        max_retries: int | None = None,
        **kwargs,
    ) -> dict:
        """HTTP request with exponential-backoff retry."""
        retries = max_retries if max_retries is not None else self.MAX_RETRIES
        session = await self._ensure_session()
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                async with session.request(method, url, **kwargs) as resp:
                    if resp.status >= 500:
                        body = await resp.text()
                        raise aiohttp.ServerConnectionError(
                            f"HTTP {resp.status}: {body[:200]}"
                        )
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < retries:
                    wait = min(2 ** attempt + random.uniform(0, 1), 30)
                    logger.warning(
                        "US API request failed [%d/%d]: %s. Retrying in %.1fs…",
                        attempt + 1, retries, e, wait,
                    )
                    await asyncio.sleep(wait)

        raise last_error  # type: ignore[misc]

    # ═════════════════════════════════════════════════════════════════
    # OAuth2 Token
    # ═════════════════════════════════════════════════════════════════

    async def _ensure_token(self) -> str:
        async with self._token_lock:
            if self._token is None or time.time() >= self._token.expires_at:
                await self._refresh_token()
        return self._token.access_token

    async def _refresh_token(self) -> None:
        url = f"{self.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.api_key,
            "appsecret": self.api_secret,
        }
        logger.info("Requesting US API OAuth2 token…")
        data = await self._request("POST", url, json=payload)
        token = data["access_token"]
        expires_in = int(data.get("expires_in", 86400))
        self._token = _TokenInfo(
            access_token=token,
            expires_at=time.time() + expires_in - self.TOKEN_REFRESH_MARGIN,
        )
        logger.info("US API token acquired (valid for %ds)", expires_in)

    # ═════════════════════════════════════════════════════════════════
    # Hashkey & Headers
    # ═════════════════════════════════════════════════════════════════

    async def _get_hashkey(self, body: dict) -> str:
        url = f"{self.base_url}/uapi/hashkey"
        headers = {
            "content-type": "application/json; charset=utf-8",
            "appkey": self.api_key,
            "appsecret": self.api_secret,
        }
        data = await self._request("POST", url, headers=headers, json=body)
        return data["HASH"]

    def _headers(self, tr_id: str, token: str) -> dict:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.api_key,
            "appsecret": self.api_secret,
            "tr_id": tr_id,
        }

    # ═════════════════════════════════════════════════════════════════
    # Public API: 해외주식 현재가 조회
    # ═════════════════════════════════════════════════════════════════

    async def get_us_price(
        self, ticker: str, exchange: str = _DEFAULT_EXCHANGE,
    ) -> float:
        """
        Fetch current price of a US stock.

        Parameters
        ----------
        ticker : str
            US stock ticker (e.g. 'AAPL', 'NVDA').
        exchange : str
            Exchange code: 'NAS' (NASDAQ), 'NYS' (NYSE), 'AMS' (AMEX).

        Returns
        -------
        float : Current price in USD.
        """
        token = await self._ensure_token()
        url = (
            f"{self.base_url}"
            "/uapi/overseas-price/v1/quotations/price"
        )
        headers = self._headers(self._tr["price"], token)
        params = {
            "AUTH": "",
            "EXCD": exchange,
            "SYMB": ticker,
        }

        data = await self._request("GET", url, headers=headers, params=params)

        if data.get("rt_cd") != "0":
            raise RuntimeError(
                f"[US:{ticker}] 현재가 조회 실패: "
                f"[{data.get('msg_cd')}] {data.get('msg1')}"
            )

        output = data["output"]
        price = float(output.get("last", "0") or "0")

        logger.debug(
            "[US:%s] price=%.2f USD, change=%s%%",
            ticker, price, output.get("rate", "?"),
        )
        return price

    # ═════════════════════════════════════════════════════════════════
    # Public API: 해외주식 주문 V2
    # ═════════════════════════════════════════════════════════════════

    async def place_us_order(
        self,
        ticker: str,
        qty: int,
        side: str,
        order_type: str = "market",
        price: float = 0,
        exchange: str = _DEFAULT_EXCHANGE,
    ) -> dict:
        """
        Place a buy or sell order for a US stock.

        Parameters
        ----------
        ticker : str
            US stock ticker (e.g. 'AAPL').
        qty : int
            Number of shares.
        side : str
            'BUY' or 'SELL'.
        order_type : str
            'market' (시장가) or 'limit' (지정가).
        price : float
            Limit price in USD (ignored for market orders).
        exchange : str
            Exchange code.

        Returns
        -------
        dict : KIS API response with order number in output.ODNO.
        """
        token = await self._ensure_token()
        url = (
            f"{self.base_url}"
            "/uapi/overseas-stock/v1/trading/order"
        )

        side_upper = side.upper()
        tr_id = self._tr["buy"] if side_upper == "BUY" else self._tr["sell"]

        # 시장가: ORD_DVSN="00" with price 0
        # 지정가: ORD_DVSN="00" with actual price
        ord_price = "0" if order_type == "market" else f"{price:.2f}"

        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "OVRS_EXCG_CD": exchange,
            "PDNO": ticker,
            "ORD_QTY": str(qty),
            "OVRS_ORD_UNPR": ord_price,
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",
        }

        hashkey = await self._get_hashkey(body)
        headers = self._headers(tr_id, token)
        headers["hashkey"] = hashkey

        order_label = "시장가" if order_type == "market" else f"지정가 ${price:.2f}"
        logger.info(
            "US 주문: %s %s x%d (%s)",
            side_upper, ticker, qty, order_label,
        )

        data = await self._request("POST", url, headers=headers, json=body)

        if data.get("rt_cd") != "0":
            raise RuntimeError(
                f"US 주문 실패: [{data.get('msg_cd')}] {data.get('msg1')}"
            )

        order_no = data.get("output", {}).get("ODNO", "N/A")
        logger.info("US 주문 접수: %s %s (주문번호=%s)", side_upper, ticker, order_no)
        return data

    # ═════════════════════════════════════════════════════════════════
    # Public API: 해외주식 잔고 조회
    # ═════════════════════════════════════════════════════════════════

    async def get_us_balance(self) -> dict:
        """
        Fetch US stock holdings and balance.

        Returns
        -------
        dict:
            {
                "total_eval_usd": float,
                "total_pnl_usd": float,
                "holdings": [
                    {
                        "ticker": str,
                        "name": str,
                        "qty": int,
                        "avg_price_usd": float,
                        "current_price_usd": float,
                        "eval_usd": float,
                        "pnl_usd": float,
                        "pnl_pct": float,
                    },
                    ...
                ]
            }
        """
        token = await self._ensure_token()
        url = (
            f"{self.base_url}"
            "/uapi/overseas-stock/v1/trading/inquire-balance"
        )
        headers = self._headers(self._tr["balance"], token)
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "OVRS_EXCG_CD": "NASD",
            "TR_CRCY_CD": "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }

        data = await self._request("GET", url, headers=headers, params=params)

        if data.get("rt_cd") != "0":
            raise RuntimeError(
                f"US 잔고 조회 실패: [{data.get('msg_cd')}] {data.get('msg1')}"
            )

        holdings = []
        for item in data.get("output1", []):
            qty = int(float(item.get("OVRS_CBLC_QTY", "0")))
            if qty <= 0:
                continue
            avg_price = float(item.get("PCHS_AVG_PRIC", "0"))
            cur_price = float(item.get("NOW_PRIC2", "0"))
            eval_amt = float(item.get("OVRS_STCK_EVLU_AMT", "0"))
            pnl = float(item.get("FRCR_EVLU_PFLS_AMT", "0"))
            pnl_pct = float(item.get("EVLU_PFLS_RT", "0"))

            holdings.append({
                "ticker": item.get("OVRS_PDNO", ""),
                "name": item.get("OVRS_ITEM_NAME", ""),
                "qty": qty,
                "avg_price_usd": avg_price,
                "current_price_usd": cur_price,
                "eval_usd": eval_amt,
                "pnl_usd": pnl,
                "pnl_pct": pnl_pct,
            })

        summary = data.get("output2", {})
        if isinstance(summary, list):
            summary = summary[0] if summary else {}

        total_eval = float(summary.get("TOT_EVLU_PFLS_AMT", "0"))
        total_pnl = float(summary.get("OVRS_TOT_PFLS", "0"))

        logger.info(
            "US 잔고: 평가=$%s, 손익=$%s, 종목수=%d",
            f"{total_eval:,.2f}", f"{total_pnl:,.2f}", len(holdings),
        )
        return {
            "total_eval_usd": total_eval,
            "total_pnl_usd": total_pnl,
            "holdings": holdings,
        }

    # ═════════════════════════════════════════════════════════════════
    # FX: 환율 조회 / 환산 유틸리티
    # ═════════════════════════════════════════════════════════════════

    async def get_exchange_rate(self) -> float:
        """
        Get current USD/KRW exchange rate.

        Tries KIS 환율 API first, falls back to cached/default value.
        Caches for 10 minutes to avoid excessive API calls.

        Returns
        -------
        float : KRW per 1 USD.
        """
        # Return cache if fresh (< 10 min)
        if time.time() - self._fx_cache_time < 600:
            return self._cached_usdkrw

        try:
            token = await self._ensure_token()
            url = (
                f"{self.base_url}"
                "/uapi/overseas-stock/v1/trading/inquire-present-balance"
            )
            headers = self._headers("CTRP6504R", token)
            params = {
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.acnt_prdt_cd,
                "WCRC_FRCR_DVSN_CD": "02",
                "NATN_CD": "840",         # US
                "TR_MKET_CD": "00",
                "INQR_DVSN_CD": "00",
            }
            data = await self._request("GET", url, headers=headers, params=params)

            if data.get("rt_cd") == "0":
                output2 = data.get("output2", [])
                if isinstance(output2, list) and output2:
                    rate = float(output2[0].get("FRST_BLTN_EXRT", "0"))
                    if rate > 0:
                        self._cached_usdkrw = rate
                        self._fx_cache_time = time.time()
                        logger.info("USD/KRW 환율: %.2f", rate)
                        return rate

        except Exception as e:
            logger.warning("환율 조회 실패: %s. 기본값 사용: %.1f", e, self.default_usdkrw)

        return self._cached_usdkrw

    def usd_to_krw(self, usd_amount: float, rate: float | None = None) -> float:
        """Convert USD to KRW using given or cached rate."""
        fx = rate or self._cached_usdkrw
        return usd_amount * fx

    def krw_to_usd(self, krw_amount: float, rate: float | None = None) -> float:
        """Convert KRW to USD using given or cached rate."""
        fx = rate or self._cached_usdkrw
        return krw_amount / fx if fx > 0 else 0.0

    # ═════════════════════════════════════════════════════════════════
    # Cleanup
    # ═════════════════════════════════════════════════════════════════

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("KIS US Client session closed.")
