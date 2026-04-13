"""
engines/forecasting/xgboost_model.py

XGBoost train and inference wrapper.
One model per (coin, horizon). Saved to models/ with date stamp.
"""

import os
import glob
import structlog
import numpy as np
import xgboost as xgb
from typing import Optional

from config.constants import PREDICTION_HORIZONS_HOURS, MODELS_DIR

logger = structlog.get_logger(__name__)

_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "verbosity": 0,
}


def train(
    X: np.ndarray, y: dict[int, np.ndarray], coin: str, date_tag: str
) -> dict[int, xgb.XGBRegressor]:
    """Train one XGBRegressor per horizon. Saves weights to models/."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    models: dict[int, xgb.XGBRegressor] = {}
    for h in PREDICTION_HORIZONS_HOURS:
        if h not in y:
            continue
        m = xgb.XGBRegressor(**_PARAMS)
        m.fit(X, y[h])
        path = os.path.join(MODELS_DIR, f"xgb_{coin.lower()}_{h}h_{date_tag}.json")
        m.save_model(path)
        logger.info("xgb_trained", coin=coin, horizon=h, path=path)
        models[h] = m
    return models


def load_latest(coin: str) -> Optional[dict[int, xgb.XGBRegressor]]:
    """Load most recently saved XGBoost models for a coin."""
    models: dict[int, xgb.XGBRegressor] = {}
    for h in PREDICTION_HORIZONS_HOURS:
        pattern = os.path.join(MODELS_DIR, f"xgb_{coin.lower()}_{h}h_*.json")
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            logger.warning("xgb_no_weights", coin=coin, horizon=h)
            return None
        m = xgb.XGBRegressor(**_PARAMS)
        m.load_model(candidates[-1])
        logger.info("xgb_loaded", coin=coin, horizon=h, path=candidates[-1])
        models[h] = m
    return models or None


def predict(models: dict[int, xgb.XGBRegressor], X_last: np.ndarray) -> dict[str, float]:
    """Inference on latest feature row."""
    x = X_last.reshape(1, -1)
    return {
        "target_24h": float(models[24].predict(x)[0]) if 24 in models else 0.0,
        "target_72h": float(models[72].predict(x)[0]) if 72 in models else 0.0,
        "target_7d": float(models[168].predict(x)[0]) if 168 in models else 0.0,
    }
