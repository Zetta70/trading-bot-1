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
    horizon: int = 10,
    pt_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    atr_period: int = 14,
    n_classes: Literal[2, 3] = 2,
) -> pd.Series:
    """
    Create a forward-looking target label using the Triple Barrier Method.

    Why not a simple forward return?
    --------------------------------
    `close_t+h / close_t - 1 >= threshold` ignores the trading path:
    a stock can spike to +5%, crash to -10%, and still close at +2.1%.
    The naive method labels that as success, but a real stop-loss trader
    would have been stopped out. Triple Barrier simulates realistic
    entry+exit, so the label reflects the actually-reachable outcome.

    Parameters
    ----------
    df : DataFrame with 'high', 'low', 'close'
    horizon : max holding period (bars)
    pt_atr_mult, sl_atr_mult : barrier widths in ATR units
    atr_period : ATR lookback
    n_classes :
      2 → binary (1 if profit target hit, else 0)
      3 → ternary (+1 profit, -1 stop, 0 time-out)

    Returns
    -------
    pd.Series of labels; NaN at the tail where the future is unknown.
    """
    from indicators import triple_barrier_labels, binary_pt_labels

    if n_classes == 3:
        tri = triple_barrier_labels(
            df, pt_atr_mult, sl_atr_mult, horizon, atr_period,
        )
        return tri.astype("Int64").rename("target")

    binary = binary_pt_labels(
        df, pt_atr_mult, sl_atr_mult, horizon, atr_period,
    )
    return binary.rename("target")


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


# ═══════════════════════════════════════════════════════════════════════
# Ensemble Predictor — LightGBM + XGBoost + CatBoost
# ═══════════════════════════════════════════════════════════════════════

class EnsemblePredictor:
    """
    Averaging ensemble of three gradient boosting libraries.

    LightGBM is fast and handles categorical well; XGBoost is robust
    on sparse features with mature regularization; CatBoost resists
    overfitting and is strong on tabular noise. Simple averaging
    reduces variance without introducing weight-tuning overfit.
    """

    def __init__(
        self,
        task: Literal["classification", "regression"] = "classification",
        seed: int = 42,
    ):
        self.task = task
        self.seed = seed
        self.models: dict[str, object] = {}
        self.feature_names: list[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        *,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> None:
        import xgboost as xgb
        from catboost import CatBoostClassifier, CatBoostRegressor

        self.feature_names = list(X_train.columns)

        # ── LightGBM ──
        lgb_params = (
            _CLASSIFICATION_PARAMS.copy()
            if self.task == "classification"
            else _REGRESSION_PARAMS.copy()
        )
        lgb_params["seed"] = self.seed
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        self.models["lgb"] = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(period=0),
            ],
        )

        # ── XGBoost ──
        xgb_params = {
            "objective": (
                "binary:logistic" if self.task == "classification"
                else "reg:squarederror"
            ),
            "eval_metric": (
                "logloss" if self.task == "classification" else "rmse"
            ),
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "seed": self.seed + 1,
            "verbosity": 0,
        }
        if self.task == "classification":
            # XGBoost auto-derives base_score from the label mean; for
            # degenerate (all-0 or all-1) training sets that would land
            # outside (0, 1) and crash. Pin it to 0.5 = neutral prior.
            mean_label = float(np.asarray(y_train).mean())
            if not (0.05 < mean_label < 0.95):
                xgb_params["base_score"] = 0.5
        dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
        dval_xgb = xgb.DMatrix(X_val, label=y_val)
        self.models["xgb"] = xgb.train(
            xgb_params,
            dtrain_xgb,
            num_boost_round=num_boost_round,
            evals=[(dval_xgb, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        # ── CatBoost ──
        cb_cls = (
            CatBoostClassifier if self.task == "classification"
            else CatBoostRegressor
        )
        self.models["cat"] = cb_cls(
            iterations=num_boost_round,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=self.seed + 2,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        self.models["cat"].fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False,
        )

        logger.info(
            "Ensemble trained: LGB best_iter=%d, XGB best_iter=%s, "
            "CB best_iter=%s",
            self.models["lgb"].best_iteration,
            getattr(self.models["xgb"], "best_iteration", "?"),
            getattr(self.models["cat"], "best_iteration_", "?"),
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        if self.task != "classification":
            raise ValueError("predict_proba is only for classification")

        p_lgb = self.models["lgb"].predict(
            X, num_iteration=self.models["lgb"].best_iteration,
        )
        p_xgb = self.models["xgb"].predict(xgb.DMatrix(X))
        p_cat = self.models["cat"].predict_proba(X)[:, 1]
        return (p_lgb + p_xgb + p_cat) / 3.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)

    def feature_importance(self) -> pd.DataFrame:
        """Averaged-rank feature importance across the three models."""
        lgb_imp = self.models["lgb"].feature_importance(importance_type="gain")
        xgb_score = self.models["xgb"].get_score(importance_type="gain")
        xgb_imp = np.array([
            xgb_score.get(name, 0.0) for name in self.feature_names
        ])
        cat_imp = self.models["cat"].get_feature_importance()

        df = pd.DataFrame({
            "feature": self.feature_names,
            "lgb": lgb_imp,
            "xgb": xgb_imp,
            "cat": cat_imp,
        })
        for c in ("lgb", "xgb", "cat"):
            df[c + "_rank"] = df[c].rank(ascending=True, pct=True)
        df["importance"] = (
            df["lgb_rank"] + df["xgb_rank"] + df["cat_rank"]
        ) / 3.0
        return df[["feature", "importance"]].sort_values(
            "importance", ascending=False,
        ).reset_index(drop=True)

    def save(self, path: str) -> None:
        """Pickle LGB/XGB state + write a sidecar .cbm for CatBoost."""
        import xgboost as xgb  # noqa: F401 (imported for type hint only)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cat_path = path.replace(".pkl", ".cbm")
        self.models["cat"].save_model(cat_path)

        payload = {
            "ensemble": True,
            "task": self.task,
            "seed": self.seed,
            "feature_names": self.feature_names,
            "lgb_model_str": self.models["lgb"].model_to_string(),
            "xgb_model_bytes": self.models["xgb"].save_raw().hex(),
            "cat_model_path": cat_path,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Ensemble saved to %s (+%s)", path, cat_path)

    def load(self, path: str) -> None:
        import xgboost as xgb
        from catboost import CatBoostClassifier, CatBoostRegressor

        with open(path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301

        self.task = payload.get("task", "classification")
        self.seed = payload.get("seed", 42)
        self.feature_names = payload["feature_names"]

        self.models["lgb"] = lgb.Booster(model_str=payload["lgb_model_str"])

        booster = xgb.Booster()
        booster.load_model(bytearray.fromhex(payload["xgb_model_bytes"]))
        self.models["xgb"] = booster

        cb_cls = (
            CatBoostClassifier if self.task == "classification"
            else CatBoostRegressor
        )
        cb_model = cb_cls()
        cb_model.load_model(payload["cat_model_path"])
        self.models["cat"] = cb_model

        logger.info("Ensemble loaded from %s", path)


# ═══════════════════════════════════════════════════════════════════════
# Meta-Labeling (Two-Stage Classifier)
# ═══════════════════════════════════════════════════════════════════════

class MetaLabeler:
    """
    Two-stage classifier for improved signal precision.

    Stage 1 (primary): trained on a permissive "is anything interesting
                       going to happen" label (high recall).
    Stage 2 (meta):    trained only on the primary-positive samples; its
                       label is the strict triple-barrier outcome.
    Final probability  = p(primary=1) * p(meta=1 | primary=1).

    The composite probability is much more conservative than a single
    classifier, so `ML_THRESHOLD_BUY` should be reduced accordingly.
    """

    def __init__(self, seed: int = 42):
        self.primary = EnsemblePredictor(task="classification", seed=seed)
        self.meta = EnsemblePredictor(task="classification", seed=seed + 100)
        self.primary_threshold: float = 0.50
        self.feature_names: list[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_primary_train: pd.Series,
        y_meta_train: pd.Series,
        X_val: pd.DataFrame,
        y_primary_val: pd.Series,
        y_meta_val: pd.Series,
    ) -> None:
        self.feature_names = list(X_train.columns)

        # Degenerate-primary guard: if every label is the same class,
        # XGBoost/CatBoost crash. Fall back to using meta as primary
        # so the pipeline still produces a usable composite probability.
        primary_unique = pd.unique(y_primary_train.dropna())
        if len(primary_unique) < 2:
            logger.warning(
                "Primary labels degenerate (single class %s). "
                "Using meta label as primary — degrades to single-stage.",
                primary_unique.tolist(),
            )
            y_primary_train = y_meta_train
            y_primary_val = y_meta_val

        logger.info("Meta-labeler: training primary model ...")
        self.primary.train(X_train, y_primary_train, X_val, y_primary_val)

        logger.info("Meta-labeler: filtering for meta training set ...")
        primary_probs = self.primary.predict_proba(X_train)
        mask = primary_probs >= self.primary_threshold

        if mask.sum() < 50:
            logger.warning(
                "Meta training set too small (%d samples). "
                "Using all samples (degrades to single-stage).",
                int(mask.sum()),
            )
            mask = np.ones(len(X_train), dtype=bool)

        X_meta = X_train.loc[mask]
        y_meta = y_meta_train.loc[mask]

        # If the filter produced single-class meta labels, widen back
        # to the full training set so the boosters can still learn.
        if len(pd.unique(y_meta.dropna())) < 2:
            logger.warning(
                "Filtered meta labels are single-class. Reverting to "
                "unfiltered training set.",
            )
            mask = np.ones(len(X_train), dtype=bool)
            X_meta = X_train
            y_meta = y_meta_train

        val_primary_probs = self.primary.predict_proba(X_val)
        val_mask = val_primary_probs >= self.primary_threshold
        if val_mask.sum() < 10:
            val_mask = np.ones(len(X_val), dtype=bool)
        if len(pd.unique(y_meta_val.loc[val_mask].dropna())) < 2:
            val_mask = np.ones(len(X_val), dtype=bool)

        logger.info(
            "Meta-labeler: training meta model on %d samples "
            "(%.1f%% of train)",
            int(mask.sum()), 100 * float(mask.sum()) / len(X_train),
        )
        self.meta.train(
            X_meta, y_meta,
            X_val.loc[val_mask], y_meta_val.loc[val_mask],
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_primary = self.primary.predict_proba(X)
        p_meta = self.meta.predict_proba(X)
        return p_primary * p_meta

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)

    def feature_importance(self) -> pd.DataFrame:
        """Feature importance from the meta model (the filter stage)."""
        return self.meta.feature_importance()

    def save(self, path: str) -> None:
        """Save primary + meta separately and leave a pointer pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        base = path.replace(".pkl", "")
        primary_path = f"{base}_primary.pkl"
        meta_path = f"{base}_meta.pkl"
        self.primary.save(primary_path)
        self.meta.save(meta_path)

        with open(path, "wb") as f:
            pickle.dump({
                "meta_labeler": True,
                "primary_path": primary_path,
                "meta_path": meta_path,
                "primary_threshold": self.primary_threshold,
                "feature_names": self.feature_names,
            }, f)
        logger.info(
            "MetaLabeler saved to %s (+%s, +%s)",
            path, primary_path, meta_path,
        )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301
        self.primary = EnsemblePredictor()
        self.primary.load(payload["primary_path"])
        self.meta = EnsemblePredictor()
        self.meta.load(payload["meta_path"])
        self.primary_threshold = payload.get("primary_threshold", 0.5)
        self.feature_names = payload.get(
            "feature_names", self.meta.feature_names,
        )
        logger.info("MetaLabeler loaded from %s", path)


# ═══════════════════════════════════════════════════════════════════════
# Primary vs Meta Labels
# ═══════════════════════════════════════════════════════════════════════

def make_primary_label(
    df: pd.DataFrame,
    horizon: int = 10,
    pt_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    threshold_atr_mult: float = 1.0,
) -> pd.Series:
    """
    Primary label: 1 if an upward move ≥ threshold_atr_mult * ATR occurs
    within horizon, 0 otherwise.

    Notes
    -----
    The spec's original 0.5 * ATR threshold is degenerate on liquid
    Korean equities (~100% positive over a 10-bar horizon). 1.0 * ATR
    keeps the primary permissive (high recall) while leaving enough
    negatives for the boosters to learn from. The label is up-only
    because the live system is long-only — a downside move isn't an
    "opportunity" we'd act on.
    """
    from indicators import atr

    atr_series = atr(df, 14)
    labels = pd.Series(index=df.index, dtype="Int64")

    closes = df["close"].values
    highs = df["high"].values
    atrs = atr_series.values
    n = len(df)

    for t in range(n - 1):
        a = atrs[t]
        if np.isnan(a) or a <= 0:
            continue
        entry = closes[t]
        up = entry + threshold_atr_mult * a
        label: object = 0
        for k in range(1, horizon + 1):
            idx = t + k
            if idx >= n:
                label = pd.NA
                break
            if highs[idx] >= up:
                label = 1
                break
        labels.iloc[t] = label

    return labels.rename("target")


def make_meta_label(
    df: pd.DataFrame,
    horizon: int = 10,
    pt_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
) -> pd.Series:
    """
    Meta label: 1 if the profit target hit FIRST (before the stop), 0
    otherwise. Strict, path-aware outcome label reused from Phase 1.
    """
    from indicators import binary_pt_labels
    return binary_pt_labels(
        df, pt_atr_mult, sl_atr_mult, horizon, 14,
    ).rename("target")
