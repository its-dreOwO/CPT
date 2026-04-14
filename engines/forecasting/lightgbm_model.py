"""
engines/forecasting/lightgbm_model.py

LightGBM train and inference wrapper. Mirrors xgboost_model.py structure.
Faster training, similar accuracy - tabular complement to XGBoost.
"""

import os
import glob
import structlog
import numpy as np
import lightgbm as lgb
from typing import Optional

from config.constants import PREDICTION_HORIZONS_HOURS, MODELS_DIR

logger = structlog.get_logger(__name__)

_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "regression",
    "device": "gpu",
    "verbose": -1,
}


def train(
    X: np.ndarray, y: dict[int, np.ndarray], coin: str, date_tag: str
) -> dict[int, lgb.LGBMRegressor]:
    """Train one LGBMRegressor per horizon. Saves weights to models/."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    models: dict[int, lgb.LGBMRegressor] = {}
    for h in PREDICTION_HORIZONS_HOURS:
        if h not in y:
            continue
        m = lgb.LGBMRegressor(**_PARAMS)
        m.fit(X, y[h])
        path = os.path.join(MODELS_DIR, f"lgbm_{coin.lower()}_{h}h_{date_tag}.txt")
        m.booster_.save_model(path)
        logger.info("lgbm_trained", coin=coin, horizon=h, path=path)
        models[h] = m
    return models


def load_latest(coin: str) -> Optional[dict[int, lgb.LGBMRegressor]]:
    """Load most recently saved LightGBM models for a coin."""
    models: dict[int, lgb.LGBMRegressor] = {}
    for h in PREDICTION_HORIZONS_HOURS:
        pattern = os.path.join(MODELS_DIR, f"lgbm_{coin.lower()}_{h}h_*.txt")
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            logger.warning("lgbm_no_weights", coin=coin, horizon=h)
            return None
        booster = lgb.Booster(model_file=candidates[-1])
        m = lgb.LGBMRegressor(**_PARAMS)
        m._Booster = booster
        logger.info("lgbm_loaded", coin=coin, horizon=h, path=candidates[-1])
        models[h] = m
    return models or None


def predict(models: dict[int, lgb.LGBMRegressor], X_last: np.ndarray) -> dict[str, float]:
    """Inference on latest feature row."""
    x = X_last.reshape(1, -1)
    return {
        "target_24h": float(models[24].predict(x)[0]) if 24 in models else 0.0,
        "target_72h": float(models[72].predict(x)[0]) if 72 in models else 0.0,
        "target_7d": float(models[168].predict(x)[0]) if 168 in models else 0.0,
    }
