"""
KIS (Korea Investment & Securities) REST API Client.

Production-ready async implementation supporting both live and paper trading.
Automatically switches server URLs and transaction IDs based on RUN_MODE.

Features:
  - OAuth2 token lifecycle with asyncio.Lock for concurrent safety
  - Exponential backoff retry on network / 5xx errors
  - Hashkey generation for POST endpoints (order security)
  - get_current_price, place_order, get_balance

API Reference: https://apiportal.koreainvestment.com/apiservice

Server URLs:
  REAL : https://openapi.koreainvestment.com:9443
  MOCK : https://openapivts.koreainvestment.com:29443  (모의투자)
"""

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

# ── Transaction IDs (실전 vs 모의 differ for trading endpoints) ──────
_TR_IDS = {
    "REAL": {
        "price":   "FHKST01010100",    # 현재가 조회
        "buy":     "TTTC0802U",         # 매수
        "sell":    "TTTC0801U",         # 매도
        "balance": "TTTC8434R",         # 잔고 조회
    },
    "MOCK": {
        "price":   "FHKST01010100",
        "buy":     "VTTC0802U",
        "sell":    "VTTC0801U",
        "balance": "VTTC8434R",
    },
}

# ── Order Type Codes ─────────────────────────────────────────────────
_ORD_DVSN = {
    "market":         "01",   # 시장가
    "limit":          "00",   # 지정가
    "conditional":    "02",   # 조건부지정가
    "best_limit":     "03",   # 최유리지정가
    "first_limit":    "04",   # 최우선지정가
    "pre_market":     "05",   # 장전 시간외
    "post_market":    "06",   # 장후 시간외
}


@dataclass
class _TokenInfo:
    """Stores OAuth2 token with its expiry timestamp."""
    access_token: str
    expires_at: float   # Unix epoch


class KISClient:
    """
    Async REST client for KIS domestic stock trading.

    Supports both live (실전) and paper (모의투자) servers via a single
    codebase. All HTTP calls use aiohttp with automatic retry logic.

    Args:
        api_key:   KIS_API_KEY from .env
        api_secret: KIS_API_SECRET from .env
        acc_no:    Account number in "XXXXXXXX-XX" format
        run_mode:  "REAL" for live trading, "MOCK" for paper trading
    """

    TOKEN_REFRESH_MARGIN = 60       # Refresh 60s before actual expiry
    MAX_RETRIES = 3                 # Retry count for transient failures
    REQUEST_TIMEOUT = 10            # Per-request timeout (seconds)

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        acc_no: str,
        run_mode: str = "MOCK",
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.run_mode = run_mode.upper()

        # Parse account: "50012345-01" → CANO="50012345", ACNT_PRDT_CD="01"
        parts = acc_no.split("-")
        self.cano = parts[0]
        self.acnt_prdt_cd = parts[1] if len(parts) > 1 else "01"

        # Server & transaction IDs based on mode
        self.base_url = _SERVERS.get(self.run_mode, _SERVERS["MOCK"])
        self._tr = _TR_IDS.get(self.run_mode, _TR_IDS["MOCK"])

        self._token: _TokenInfo | None = None
        self._session: aiohttp.ClientSession | None = None
        self._token_lock = asyncio.Lock()

        mode_label = "실전투자" if self.run_mode == "REAL" else "모의투자"
        logger.info(
            "KIS Client [%s] → %s (계좌: %s-%s)",
            mode_label, self.base_url, self.cano, self.acnt_prdt_cd,
        )

    # ═════════════════════════════════════════════════════════════════
    #  Session & Retry Infrastructure
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
        """
        Execute an HTTP request with exponential-backoff retry.

        Retries on:
          - aiohttp.ClientError (network failures)
          - asyncio.TimeoutError
          - HTTP 5xx (server errors)

        Does NOT retry on 4xx (client errors — bad request, auth, etc.).
        """
        retries = max_retries if max_retries is not None else self.MAX_RETRIES
        session = await self._ensure_session()
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                async with session.request(method, url, **kwargs) as resp:
                    # 5xx → retry
                    if resp.status >= 500:
                        body = await resp.text()
                        raise aiohttp.ServerConnectionError(
                            f"HTTP {resp.status}: {body[:200]}"
                        )
                    resp.raise_for_status()
                    return await resp.json()

            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
            ) as e:
                last_error = e
                if attempt < retries:
                    wait = min(2 ** attempt + random.uniform(0, 1), 30)
                    logger.warning(
                        "Request failed [%d/%d]: %s. Retrying in %.1fs…",
                        attempt + 1, retries, e, wait,
                    )
                    await asyncio.sleep(wait)

        raise last_error  # type: ignore[misc]

    # ═════════════════════════════════════════════════════════════════
    #  OAuth2 Token Management
    # ═════════════════════════════════════════════════════════════════

    async def _ensure_token(self) -> str:
        """Return a valid access token, refreshing under lock if expired."""
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

        logger.info("Requesting new OAuth2 access token…")
        data = await self._request("POST", url, json=payload)

        token = data["access_token"]
        expires_in = int(data.get("expires_in", 86400))
        self._token = _TokenInfo(
            access_token=token,
            expires_at=time.time() + expires_in - self.TOKEN_REFRESH_MARGIN,
        )
        logger.info("Token acquired (valid for %ds)", expires_in)

    # ═════════════════════════════════════════════════════════════════
    #  Hashkey (POST 요청 보안 해시)
    # ═════════════════════════════════════════════════════════════════

    async def _get_hashkey(self, body: dict) -> str:
        """
        Generate a hashkey required for POST trading endpoints.

        KIS requires this hash header to verify the request body
        has not been tampered with in transit.
        """
        url = f"{self.base_url}/uapi/hashkey"
        headers = {
            "content-type": "application/json; charset=utf-8",
            "appkey": self.api_key,
            "appsecret": self.api_secret,
        }
        data = await self._request("POST", url, headers=headers, json=body)
        return data["HASH"]

    # ═════════════════════════════════════════════════════════════════
    #  Header Builder
    # ═════════════════════════════════════════════════════════════════

    def _headers(self, tr_id: str, token: str) -> dict:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.api_key,
            "appsecret": self.api_secret,
            "tr_id": tr_id,
        }

    # ═════════════════════════════════════════════════════════════════
    #  Public API: 현재가 조회
    # ═════════════════════════════════════════════════════════════════

    async def get_current_price(self, ticker: str) -> int:
        """
        Fetch the current price of a KOSPI/KOSDAQ stock.

        Args:
            ticker: 6-digit stock code (e.g. "005930" 삼성전자)

        Returns:
            Current price in KRW (integer).

        Raises:
            RuntimeError: If the KIS API returns an error response.
        """
        token = await self._ensure_token()
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = self._headers(self._tr["price"], token)
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",   # J = KOSPI/KOSDAQ
            "FID_INPUT_ISCD": ticker,
        }

        data = await self._request("GET", url, headers=headers, params=params)

        if data.get("rt_cd") != "0":
            raise RuntimeError(
                f"[{ticker}] 현재가 조회 실패: "
                f"[{data.get('msg_cd')}] {data.get('msg1')}"
            )

        output = data["output"]
        price = int(output["stck_prpr"])  # 주식 현재가

        logger.debug(
            "[%s] 현재가=%d, 전일대비=%s, 거래량=%s",
            ticker, price,
            output.get("prdy_vrss", "?"),
            output.get("acml_vol", "?"),
        )
        return price

    # ═════════════════════════════════════════════════════════════════
    #  Public API: 주문 (매수/매도)
    # ═════════════════════════════════════════════════════════════════

    async def place_order(
        self,
        ticker: str,
        qty: int,
        side: str,
        order_type: str = "market",
        price: int = 0,
    ) -> dict:
        """
        Place a buy or sell order.

        Args:
            ticker:     6-digit stock code
            qty:        Number of shares
            side:       "BUY" or "SELL"
            order_type: "market" (시장가) | "limit" (지정가) | etc.
            price:      Limit price in KRW (required for limit orders,
                        ignored for market orders)

        Returns:
            KIS API response dict.  Order number is in
            result["output"]["ODNO"].

        Raises:
            RuntimeError: If the order is rejected by KIS.
        """
        token = await self._ensure_token()
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"

        side_upper = side.upper()
        tr_id = self._tr["buy"] if side_upper == "BUY" else self._tr["sell"]

        ord_dvsn = _ORD_DVSN.get(order_type, "01")
        # 시장가 주문은 가격을 0으로 전송
        ord_price = 0 if order_type == "market" else price

        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": ticker,                     # 종목코드
            "ORD_DVSN": ord_dvsn,               # 주문구분
            "ORD_QTY": str(qty),                # 주문수량 (문자열)
            "ORD_UNPR": str(ord_price),          # 주문단가
        }

        # POST 주문 요청은 hashkey 필요
        hashkey = await self._get_hashkey(body)
        headers = self._headers(tr_id, token)
        headers["hashkey"] = hashkey

        order_label = "시장가" if order_type == "market" else f"지정가 {price:,}"
        logger.info(
            "주문 요청: %s %s x%d (%s)",
            side_upper, ticker, qty, order_label,
        )

        data = await self._request("POST", url, headers=headers, json=body)

        if data.get("rt_cd") != "0":
            raise RuntimeError(
                f"주문 실패: [{data.get('msg_cd')}] {data.get('msg1')}"
            )

        order_no = data.get("output", {}).get("ODNO", "N/A")
        logger.info("주문 접수: %s %s (주문번호=%s)", side_upper, ticker, order_no)
        return data

    # ═════════════════════════════════════════════════════════════════
    #  Public API: 잔고 조회
    # ═════════════════════════════════════════════════════════════════

    async def get_balance(self) -> dict:
        """
        Fetch account balance: deposit, holdings, and total valuation.

        Returns:
            {
                "deposit": int,         # 예수금 (KRW)
                "total_eval": int,      # 총 평가금액
                "total_pnl": int,       # 평가손익 합계
                "pnl_pct": float,       # 평가수익률 (%)
                "holdings": [
                    {
                        "ticker": str,
                        "name": str,            # 종목명
                        "qty": int,             # 보유수량
                        "avg_price": int,       # 매입평균가
                        "current_price": int,   # 현재가
                        "eval_amount": int,     # 평가금액
                        "pnl": int,             # 평가손익
                        "pnl_pct": float,       # 수익률 (%)
                    },
                    ...
                ]
            }
        """
        token = await self._ensure_token()
        url = (
            f"{self.base_url}"
            "/uapi/domestic-stock/v1/trading/inquire-balance"
        )
        headers = self._headers(self._tr["balance"], token)
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",               # 시간외단일가여부
            "OFL_YN": "",                       # 오프라인여부
            "INQR_DVSN": "02",                  # 조회구분: 종목별
            "UNPR_DVSN": "01",                  # 단가구분
            "FUND_STTL_ICLD_YN": "N",           # 펀드결제포함여부
            "FNCG_AMT_AUTO_RDPT_YN": "N",       # 융자금액자동상환여부
            "PRCS_DVSN": "01",                  # 처리구분
            "CTX_AREA_FK100": "",               # 연속조회검색조건
            "CTX_AREA_NK100": "",               # 연속조회키
        }

        data = await self._request("GET", url, headers=headers, params=params)

        if data.get("rt_cd") != "0":
            raise RuntimeError(
                f"잔고 조회 실패: [{data.get('msg_cd')}] {data.get('msg1')}"
            )

        # ── Parse holdings (output1) ─────────────────────────────
        holdings = []
        for item in data.get("output1", []):
            qty = int(item.get("HLDG_QTY", "0"))
            if qty <= 0:
                continue
            holdings.append({
                "ticker": item["PDNO"],
                "name": item.get("PRDT_NAME", ""),
                "qty": qty,
                "avg_price": int(float(item.get("PCHS_AVG_PRIC", "0"))),
                "current_price": int(item.get("PRPR", "0")),
                "eval_amount": int(item.get("EVLU_AMT", "0")),
                "pnl": int(item.get("EVLU_PFLS_AMT", "0")),
                "pnl_pct": float(item.get("EVLU_PFLS_RT", "0")),
            })

        # ── Parse summary (output2) ─────────────────────────────
        summary = data.get("output2", [{}])
        if isinstance(summary, list):
            summary = summary[0] if summary else {}

        deposit = int(summary.get("DNCA_TOT_AMT", "0"))
        total_eval = int(summary.get("TOT_EVLU_AMT", "0"))
        total_pnl = int(summary.get("EVLU_PFLS_SMTL_AMT", "0"))
        purchase_total = int(summary.get("PCHS_AMT_SMTL_AMT", "0"))
        pnl_pct = (
            (total_pnl / purchase_total * 100) if purchase_total else 0.0
        )

        result = {
            "deposit": deposit,
            "total_eval": total_eval,
            "total_pnl": total_pnl,
            "pnl_pct": round(pnl_pct, 2),
            "holdings": holdings,
        }

        logger.info(
            "잔고 조회: 예수금=%s, 평가=%s, 손익=%s (%+.2f%%), 종목수=%d",
            f"{deposit:,}", f"{total_eval:,}",
            f"{total_pnl:,}", pnl_pct, len(holdings),
        )
        return result

    # ═════════════════════════════════════════════════════════════════
    #  Cleanup
    # ═════════════════════════════════════════════════════════════════

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("KIS Client session closed.")
