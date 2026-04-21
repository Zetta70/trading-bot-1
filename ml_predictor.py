"""
ML-based Prediction Engine.

Dual-mode predictor:
  1. Legacy mode (no trained model) — rule-based bullish score (0.0–1.0)
     using RSI/SMA/Volatility, for backward compatibility with trading_bot.py.
  2. ML mode (trained LightGBM model) — predict_signal() returning
     BUY/SELL/HOLD based on probability threshold.

The existing trading_bot.py calls:
    predictor.score(prices: list[int]) -> float
    MLPredictor.MIN_DATA  (class attribute)

Both are preserved so the live bot works without any changes.
"""

from __future__ import annotations

import math
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Unified predictor supporting both legacy scoring and LightGBM inference.

    Parameters
    ----------
    model_path : str, optional
        Path to a saved StockPredictor pickle. If provided and the file
        exists, ML mode is activated for predict_signal().
    threshold_buy : float
        P(up) threshold for BUY in ML mode.
    threshold_sell : float
        P(up) threshold below which SELL in ML mode.
    """

    # ── Class-level constants (backward compat) ──────────────────
    RSI_PERIOD = 14
    SMA_PERIOD = 20
    VOL_PERIOD = 60
    MIN_DATA = RSI_PERIOD + 1  # 15 ticks — used by portfolio_manager.py

    def __init__(
        self,
        model_path: str | None = None,
        threshold_buy: float = 0.55,
        threshold_sell: float = 0.45,
    ):
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self._ml_model = None

        if model_path and Path(model_path).exists():
            try:
                from model import StockPredictor
                self._ml_model = StockPredictor(task="classification")
                self._ml_model.load(model_path)
                logger.info("ML model loaded from %s", model_path)
            except Exception as e:
                logger.warning(
                    "Failed to load ML model from %s: %s. "
                    "Falling back to rule-based scoring.",
                    model_path, e,
                )
                self._ml_model = None

    @classmethod
    def from_config(cls) -> "MLPredictor":
        """
        Create an MLPredictor with thresholds from .env (via Config).

        This ensures both live operation and backtest use the same
        ML_THRESHOLD_BUY / ML_THRESHOLD_SELL / ML_MODEL_PATH values
        from the .env file.
        """
        from config import Config
        cfg = Config()
        return cls(
            model_path=cfg.ml_model_path,
            threshold_buy=cfg.ml_threshold_buy,
            threshold_sell=cfg.ml_threshold_sell,
        )

    # ═══════════════════════════════════════════════════════════════
    # Legacy API — used by trading_bot.py (DO NOT change signature)
    # ═══════════════════════════════════════════════════════════════

    def calculate_features(self, prices: list[int]) -> dict | None:
        """Extract technical features from a price series."""
        if len(prices) < self.MIN_DATA:
            return None

        current = prices[-1]
        sma_len = min(len(prices), self.SMA_PERIOD)
        sma = sum(prices[-sma_len:]) / sma_len

        return {
            "price": current,
            "sma": sma,
            "sma_deviation": (current - sma) / sma * 100,
            "rsi": self._rsi(prices),
            "volatility": self._volatility(prices),
        }

    def score(self, prices: list[int]) -> float:
        """
        Compute the bullish score (0.0–1.0).

        Returns 0.5 (neutral) during warmup when data is insufficient.
        """
        features = self.calculate_features(prices)
        if features is None:
            return 0.5

        return self._bullish_score(
            features["rsi"],
            features["sma_deviation"],
            features["volatility"],
        )

    # ═══════════════════════════════════════════════════════════════
    # New ML API — for enhanced signal generation
    # ═══════════════════════════════════════════════════════════════

    def predict_signal(
        self,
        ohlcv_df: pd.DataFrame,
        index_df: pd.DataFrame | None = None,
    ) -> str:
        """
        Generate a trading signal using the LightGBM model.

        Falls back to rule-based scoring if no ML model is loaded.

        Parameters
        ----------
        ohlcv_df : pd.DataFrame
            OHLCV data with columns: open, high, low, close, volume.
        index_df : pd.DataFrame, optional
            KOSPI index data for market-context features.

        Returns
        -------
        str : 'BUY', 'SELL', or 'HOLD'
        """
        if self._ml_model is None:
            # Fallback: use legacy score
            prices = ohlcv_df["close"].tolist()
            s = self.score(prices)
            if s >= self.threshold_buy:
                return "BUY"
            elif s <= self.threshold_sell:
                return "SELL"
            return "HOLD"

        try:
            from features import add_features, feature_columns

            has_market = index_df is not None
            features_df = add_features(ohlcv_df, index_df)
            cols = feature_columns(include_market=has_market)

            # Use only the columns the model was trained on
            model_cols = self._ml_model.feature_names
            if model_cols:
                available = [c for c in model_cols if c in features_df.columns]
            else:
                available = [c for c in cols if c in features_df.columns]

            latest = features_df[available].iloc[[-1]]

            # Skip if too many NaN features
            non_null = latest.dropna(axis=1)
            if len(non_null.columns) < len(available) * 0.5:
                logger.debug("Too many NaN features, returning HOLD")
                return "HOLD"

            # Fill remaining NaN with 0 for inference
            latest = latest.fillna(0)
            prob = self._ml_model.predict_proba(latest)[0]

            if prob >= self.threshold_buy:
                return "BUY"
            elif prob <= self.threshold_sell:
                return "SELL"
            return "HOLD"

        except Exception as e:
            logger.error("ML prediction failed: %s. Falling back.", e)
            prices = ohlcv_df["close"].tolist()
            s = self.score(prices)
            if s >= self.threshold_buy:
                return "BUY"
            elif s <= self.threshold_sell:
                return "SELL"
            return "HOLD"

    # ═══════════════════════════════════════════════════════════════
    # Internal: Legacy feature calculations
    # ═══════════════════════════════════════════════════════════════

    def _rsi(self, prices: list[int]) -> float:
        """Classic RSI with simple averaging over the last RSI_PERIOD."""
        deltas = [
            prices[i] - prices[i - 1]
            for i in range(max(1, len(prices) - self.RSI_PERIOD), len(prices))
        ]
        gains = sum(d for d in deltas if d > 0)
        losses = sum(-d for d in deltas if d < 0)

        avg_gain = gains / self.RSI_PERIOD
        avg_loss = losses / self.RSI_PERIOD

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _volatility(self, prices: list[int]) -> float:
        """Volatility of returns (std dev)."""
        n = min(len(prices), self.VOL_PERIOD + 1)
        if n < 3:
            return 0.0

        segment = prices[-n:]
        returns = [
            (segment[i] - segment[i - 1]) / segment[i - 1]
            for i in range(1, len(segment))
        ]
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    def _bullish_score(
        self, rsi: float, sma_dev: float, volatility: float,
    ) -> float:
        """
        Weighted combination of three sub-scores.

        Weights: RSI 40%, SMA deviation 35%, Volatility 25%.
        """
        # RSI sub-score: peaks around 55–65
        if 50 <= rsi <= 70:
            rsi_s = 1.0 - abs(rsi - 60) / 20
        elif 30 <= rsi < 50:
            rsi_s = (rsi - 30) / 40
        elif rsi > 70:
            rsi_s = max(0.0, (100 - rsi) / 30)
        else:
            rsi_s = 0.1

        # SMA deviation sub-score
        sma_s = min(1.0, max(0.0, (sma_dev + 0.5) / 1.5))

        # Volatility sub-score
        vol_pct = volatility * 100
        if vol_pct < 0.2:
            vol_s = 0.6
        elif vol_pct <= 1.0:
            vol_s = 1.0
        elif vol_pct <= 2.5:
            vol_s = max(0.2, 1.0 - (vol_pct - 1.0) / 2.0)
        else:
            vol_s = 0.1

        return 0.40 * rsi_s + 0.35 * sma_s + 0.25 * vol_s
