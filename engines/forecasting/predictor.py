"""
engines/forecasting/predictor.py

Inference entry point for the ML forecasting engine.

Loads all model weights, builds the feature matrix, runs inference
across all 5 models, and returns a PredictionResult via ensemble.py.

Called by:
  - pipeline/prediction_pipeline.py  (production)
  - python -m engines.forecasting.predictor  (manual smoke test)

Rules:
  - NEVER retrain here. Load weights only.
  - If any model fails, log and skip — ensemble handles missing models.
  - Always use the latest dated weights for each coin.
"""

from __future__ import annotations

import structlog
import numpy as np
import pandas as pd
from typing import Optional

from config.constants import SEQUENCE_LENGTH
from engines.onchain.onchain_aggregator import OnChainSnapshot
from engines.forecasting.feature_builder import build_features
from engines.forecasting.ensemble import PredictionResult, combine

# Individual model modules
from engines.forecasting import (
    timesfm_model,
    xgboost_model,
    lightgbm_model,
    lstm_model,
    transformer_model,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_timesfm(price_df: pd.DataFrame) -> Optional[dict[str, float]]:
    try:
        close = price_df["close"].to_numpy(dtype=np.float32)
        return timesfm_model.predict(close)
    except Exception as exc:
        logger.warning("predictor_timesfm_failed", error=str(exc))
        return None


def _run_xgboost(coin: str, feature_df: pd.DataFrame) -> Optional[dict[str, float]]:
    try:
        models = xgboost_model.load_latest(coin)
        if models is None:
            logger.info("predictor_xgboost_no_weights", coin=coin)
            return None
        X_last = feature_df.iloc[-1].to_numpy(dtype=np.float64)
        return xgboost_model.predict(models, X_last)
    except Exception as exc:
        logger.warning("predictor_xgboost_failed", error=str(exc))
        return None


def _run_lightgbm(coin: str, feature_df: pd.DataFrame) -> Optional[dict[str, float]]:
    try:
        models = lightgbm_model.load_latest(coin)
        if models is None:
            logger.info("predictor_lightgbm_no_weights", coin=coin)
            return None
        X_last = feature_df.iloc[-1].to_numpy(dtype=np.float64)
        return lightgbm_model.predict(models, X_last)
    except Exception as exc:
        logger.warning("predictor_lightgbm_failed", error=str(exc))
        return None


def _run_lstm(coin: str, feature_df: pd.DataFrame) -> Optional[dict[str, float]]:
    try:
        model = lstm_model.load_latest(coin)
        if model is None:
            logger.info("predictor_lstm_no_weights", coin=coin)
            return None
        seq_len = SEQUENCE_LENGTH * 24  # 60 days × 24 hours
        arr = feature_df.tail(seq_len).to_numpy(dtype=np.float32)
        if len(arr) < seq_len:
            # Pad front with zeros if history is shorter than seq_len
            pad = np.zeros((seq_len - len(arr), arr.shape[1]), dtype=np.float32)
            arr = np.vstack([pad, arr])
        return lstm_model.predict(model, arr)
    except Exception as exc:
        logger.warning("predictor_lstm_failed", error=str(exc))
        return None


def _run_tft(
    coin: str, feature_df: pd.DataFrame, price_df: pd.DataFrame
) -> Optional[dict[str, float]]:
    try:
        model = transformer_model.load_latest(coin)
        if model is None:
            logger.info("predictor_tft_no_weights", coin=coin)
            return None
        # TFT needs the full feature_df with a 'close' column
        df = feature_df.copy()
        if "close" not in df.columns and "close" in price_df.columns:
            df["close"] = price_df["close"].reindex(df.index).ffill().fillna(0.0)
        return transformer_model.predict(model, df, coin)
    except Exception as exc:
        logger.warning("predictor_tft_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    coin: str,
    price_df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    sentiment_df: Optional[pd.DataFrame] = None,
    onchain: Optional[OnChainSnapshot] = None,
) -> PredictionResult:
    """Run full multi-model inference and return an ensemble PredictionResult.

    Args:
        coin:         'SOL' or 'DOGE'
        price_df:     Hourly OHLCV DataFrame with UTC DatetimeIndex.
                      Must have at least `close` column. Minimum 24 rows.
        macro_df:     Optional macro features from macro_features.build_features().
        sentiment_df: Optional sentiment features from sentiment_features.build_features().
        onchain:      Optional latest OnChainSnapshot for the coin.

    Returns:
        PredictionResult — ensemble of all available model outputs.
        Falls back to last known price if all models fail.
    """
    if price_df.empty or len(price_df) < 24:
        logger.warning("predictor_insufficient_history", coin=coin, rows=len(price_df))
        last = float(price_df["close"].iloc[-1]) if not price_df.empty else 0.0
        return _fallback_result(coin, last)

    current_price = float(price_df["close"].iloc[-1])
    logger.info("predictor_start", coin=coin, current_price=current_price, rows=len(price_df))

    # Build unified feature matrix (all signals merged)
    feature_df = build_features(price_df, macro_df, sentiment_df, onchain)
    if feature_df.empty:
        logger.warning("predictor_feature_build_failed", coin=coin)
        return _fallback_result(coin, current_price)

    # Run all models — failures are isolated, non-fatal
    model_outputs: dict[str, dict[str, float]] = {}

    tfm = _run_timesfm(price_df)
    if tfm:
        model_outputs["timesfm"] = tfm

    xgb = _run_xgboost(coin, feature_df)
    if xgb:
        model_outputs["xgboost"] = xgb

    lgbm = _run_lightgbm(coin, feature_df)
    if lgbm:
        model_outputs["lightgbm"] = lgbm

    lstm = _run_lstm(coin, feature_df)
    if lstm:
        model_outputs["lstm"] = lstm

    tft = _run_tft(coin, feature_df, price_df)
    if tft:
        model_outputs["tft"] = tft

    logger.info(
        "predictor_models_run",
        coin=coin,
        models_available=list(model_outputs.keys()),
    )

    if not model_outputs:
        logger.error("predictor_all_models_failed", coin=coin)
        return _fallback_result(coin, current_price)

    return combine(coin, current_price, model_outputs)


def _fallback_result(coin: str, current_price: float) -> PredictionResult:
    """Return a zero-confidence result at the last known price."""
    return PredictionResult(
        coin=coin,
        current_price=current_price,
        target_24h=current_price,
        target_72h=current_price,
        target_7d=current_price,
        pct_change_24h=0.0,
        pct_change_72h=0.0,
        pct_change_7d=0.0,
        direction_24h="FLAT",
        confidence=0.0,
        model_outputs={},
        models_used=[],
    )


# ---------------------------------------------------------------------------
# Smoke test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from config.logging_config import setup_logging

    setup_logging()

    coin = sys.argv[1] if len(sys.argv) > 1 else "SOL"

    # Generate synthetic price series for smoke test (no live data needed)
    np.random.seed(42)
    n = 500
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.002, n)))
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="1h", tz="UTC")
    price_df = pd.DataFrame(
        {
            "open": prices * (1 - 0.001),
            "high": prices * 1.003,
            "low": prices * 0.997,
            "close": prices,
            "volume": np.random.uniform(1e5, 1e6, n),
        },
        index=idx,
    )

    result = run(coin=coin, price_df=price_df)

    print(f"\n=== Predictor smoke test: {coin} ===")
    print(f"Current price : {result.current_price:.6f}")
    print(f"Target 24h    : {result.target_24h:.6f}  ({result.pct_change_24h:+.2f}%)")
    print(f"Target 72h    : {result.target_72h:.6f}  ({result.pct_change_72h:+.2f}%)")
    print(f"Target 7d     : {result.target_7d:.6f}  ({result.pct_change_7d:+.2f}%)")
    print(f"Direction     : {result.direction_24h}")
    print(f"Confidence    : {result.confidence:.4f}")
    print(f"Models used   : {result.models_used}")
    print(f"Should alert  : {result.should_alert()}")
