"""
LightGBM Model Wrapper for Stock Prediction.

Provides a unified interface for training, prediction, feature importance,
and model persistence. Supports both classification (buy/hold probability)
and regression (expected return) tasks.

Usage
-----
    from model import StockPredictor, make_target

    predictor = StockPredictor(task="classification")
    predictor.train(X_train, y_train, X_val, y_val)
    prob = predictor.predict_proba(X_new)  # P(up)
    predictor.save("models/lgbm_v1.pkl")
"""

from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Default hyper-parameters (anti-overfit oriented)
# ═══════════════════════════════════════════════════════════════════════

_CLASSIFICATION_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
}

_REGRESSION_PARAMS: dict = {
    **_CLASSIFICATION_PARAMS,
    "objective": "regression",
    "metric": "rmse",
}


# ═══════════════════════════════════════════════════════════════════════
# Target Label Factory
# ═══════════════════════════════════════════════════════════════════════

def make_target(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.02,
    n_classes: Literal[2, 3] = 2,
) -> pd.Series:
    """
    Create a forward-looking target label.

    The target is based on the return from close at time t to close at
    time t + horizon.  This uses shift(-horizon), which is **only safe
    inside the backtesting pipeline** — never in live feature generation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'close' column.
    horizon : int
        Number of trading days into the future.
    threshold : float
        Return threshold for labelling.
    n_classes : {2, 3}
        2 → binary (1 if ret ≥ +threshold, else 0).
        3 → ternary (1 if ret ≥ +threshold, -1 if ret ≤ -threshold, else 0).

    Returns
    -------
    pd.Series
        Integer labels. NaN at the tail where the forward return is unknown.
    """
    future_ret = df["close"].pct_change(horizon).shift(-horizon)

    if n_classes == 2:
        target = (future_ret >= threshold).astype("Int64")
    else:
        target = pd.Series(0, index=df.index, dtype="Int64")
        target[future_ret >= threshold] = 1
        target[future_ret <= -threshold] = -1

    # Mark rows where future return is unknown as NA
    target[future_ret.isna()] = pd.NA
    target.name = "target"
    return target


# ═══════════════════════════════════════════════════════════════════════
# Model Wrapper
# ═══════════════════════════════════════════════════════════════════════

class StockPredictor:
    """
    LightGBM wrapper for stock prediction.

    Parameters
    ----------
    task : {'classification', 'regression'}
    params : dict, optional
        Custom LightGBM parameters. Merged on top of defaults.
    """

    def __init__(
        self,
        task: Literal["classification", "regression"] = "classification",
        params: dict | None = None,
    ):
        self.task = task
        base = (
            _CLASSIFICATION_PARAMS.copy()
            if task == "classification"
            else _REGRESSION_PARAMS.copy()
        )
        if params:
            base.update(params)
        self.params = base
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []

    # ── Training ─────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        *,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> None:
        """
        Train the LightGBM model with early stopping.

        Parameters
        ----------
        X_train, y_train : Training data.
        X_val, y_val : Validation data for early stopping.
        num_boost_round : Maximum boosting iterations.
        early_stopping_rounds : Stop if val metric doesn't improve.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ]

        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
        logger.info(
            "Training complete. Best iteration: %d",
            self.model.best_iteration,
        )

    # ── Prediction ───────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict P(label=1) for classification task.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if self.task != "classification":
            raise ValueError("predict_proba is only for classification task")
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict target value (regression) or raw score (classification).

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    # ── Feature Importance ───────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance ranked by gain.

        Returns
        -------
        pd.DataFrame with columns ['feature', 'importance']
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        importance = self.model.feature_importance(importance_type="gain")
        names = (
            self.feature_names
            if self.feature_names
            else [f"f{i}" for i in range(len(importance))]
        )
        result = pd.DataFrame({
            "feature": names,
            "importance": importance,
        })
        return result.sort_values("importance", ascending=False).reset_index(
            drop=True,
        )

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the model, params, and feature names to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": self.task,
            "params": self.params,
            "feature_names": self.feature_names,
            "model_str": self.model.model_to_string() if self.model else None,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved model."""
        with open(path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301
        self.task = payload["task"]
        self.params = payload["params"]
        self.feature_names = payload["feature_names"]
        if payload["model_str"]:
            self.model = lgb.Booster(model_str=payload["model_str"])
        logger.info("Model loaded from %s", path)
