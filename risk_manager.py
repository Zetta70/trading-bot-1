"""
Portfolio-level Risk Management.

Layered risk system:
  1. Per-stock stop (handled by the bot/simulator)
  2. Sector exposure cap
  3. Rolling VaR on portfolio returns
  4. Regime-based deployment scaling
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Three-layer portfolio risk controller.

    Layer 1: Sector exposure cap
    Layer 2: Rolling 60-day VaR(95%) of portfolio
    Layer 3: Regime-based deployment multiplier
             (VIX/VKOSPI + drawdown driven)
    """

    def __init__(
        self,
        max_sector_exposure: float = 0.30,
        daily_var_limit: float = 0.03,
        portfolio_stop_loss: float = -0.05,
        portfolio_kill: float = -0.10,
        vol_index_high: float = 28.0,
        vol_index_low: float = 15.0,
    ):
        self.max_sector_exposure = max_sector_exposure
        self.daily_var_limit = daily_var_limit
        self.portfolio_stop_loss = portfolio_stop_loss
        self.portfolio_kill = portfolio_kill
        self.vol_index_high = vol_index_high
        self.vol_index_low = vol_index_low

        self._sector_map: dict[str, str] = {}
        self._daily_returns_buffer: list[float] = []

    def register_sector(self, ticker: str, sector: str) -> None:
        self._sector_map[ticker] = sector

    def check_sector_concurrence(
        self,
        candidate_ticker: str,
        existing_positions: dict[str, float],
        portfolio_equity: float,
    ) -> bool:
        cand_sector = self._sector_map.get(candidate_ticker, "UNKNOWN")
        if cand_sector == "UNKNOWN":
            return True

        sector_value = sum(
            value for ticker, value in existing_positions.items()
            if self._sector_map.get(ticker) == cand_sector
        )
        ratio = sector_value / portfolio_equity if portfolio_equity > 0 else 0
        return ratio < self.max_sector_exposure

    def compute_deployment_scale(
        self,
        current_vol_index: float | None,
        current_drawdown: float,
    ) -> float:
        """Scale multiplier in [0, 1] for new position sizes."""
        if current_drawdown <= self.portfolio_kill:
            logger.critical(
                "Portfolio kill: DD=%.2f%%", current_drawdown * 100,
            )
            return 0.0
        if current_drawdown <= self.portfolio_stop_loss:
            logger.warning(
                "Portfolio stop: DD=%.2f%% — scaling down 50%%",
                current_drawdown * 100,
            )
            dd_scale = 0.5
        else:
            dd_scale = 1.0

        if current_vol_index is None or np.isnan(current_vol_index):
            vol_scale = 1.0
        elif current_vol_index >= self.vol_index_high:
            vol_scale = 0.6
        elif current_vol_index <= self.vol_index_low:
            vol_scale = 1.0
        else:
            r = (current_vol_index - self.vol_index_low) / (
                self.vol_index_high - self.vol_index_low
            )
            vol_scale = 1.0 - r * 0.4

        return min(dd_scale, vol_scale)

    def compute_var(
        self,
        return_series: pd.Series,
        alpha: float = 0.05,
    ) -> float:
        rets = return_series.dropna().tail(60)
        if len(rets) < 20:
            return 0.0
        return float(-np.percentile(rets, alpha * 100))

    def update_daily_return(self, daily_ret: float) -> None:
        self._daily_returns_buffer.append(daily_ret)
        if len(self._daily_returns_buffer) > 252:
            self._daily_returns_buffer = self._daily_returns_buffer[-252:]

    def is_var_breached(self) -> bool:
        if len(self._daily_returns_buffer) < 20:
            return False
        rets = pd.Series(self._daily_returns_buffer)
        var = self.compute_var(rets, alpha=0.05)
        return var > self.daily_var_limit
