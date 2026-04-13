"""
engines/forecasting/timesfm_model.py

TimesFM 2.5 zero-shot inference wrapper.
Model: google/timesfm-2.5-200m-pytorch (~800MB, auto-downloaded on first run).
Zero-shot: no training needed. Returns point forecasts for 24h/72h/7d.
GPU required for reasonable latency. Falls back gracefully.
"""

import structlog
import numpy as np

from config.constants import (
    TIMESFM_MODEL_ID,
    TIMESFM_CONTEXT_LEN,
    TIMESFM_HORIZON_LEN,
    PREDICTION_HORIZONS_HOURS,
)

logger = structlog.get_logger(__name__)
_model = None


def _get_model():
    """Lazy-load TimesFM 2.5 (downloads ~800MB on first call)."""
    global _model
    if _model is None:
        import timesfm

        logger.info("timesfm_loading", model=TIMESFM_MODEL_ID)
        # TimesFM 2.5 API: from_pretrained -> compile(ForecastConfig) -> forecast()
        _model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_MODEL_ID)
        _model.compile(
            timesfm.ForecastConfig(
                max_context=TIMESFM_CONTEXT_LEN,
                max_horizon=TIMESFM_HORIZON_LEN,
                per_core_batch_size=32,
            )
        )
        logger.info("timesfm_loaded")
    return _model


def predict(close_prices: np.ndarray) -> dict[str, float]:
    """Run zero-shot TimesFM inference on hourly close price series.

    Args:
        close_prices: 1-D float array, oldest first. Truncated to TIMESFM_CONTEXT_LEN.

    Returns:
        Dict with target_24h, target_72h, target_7d. Falls back to last price on error.
    """
    if len(close_prices) == 0:
        logger.warning("timesfm_empty_input")
        return {"target_24h": 0.0, "target_72h": 0.0, "target_7d": 0.0}

    context = close_prices[-TIMESFM_CONTEXT_LEN:]
    last_price = float(context[-1])
    try:
        # API: forecast(horizon=int, inputs=list[np.ndarray]) -> (point, quantile)
        point_forecasts, _ = _get_model().forecast(
            horizon=TIMESFM_HORIZON_LEN,
            inputs=[context.astype(np.float32)],
        )
        fc = point_forecasts[0]
        result = {
            "target_24h": float(fc[PREDICTION_HORIZONS_HOURS[0] - 1]),
            "target_72h": float(fc[PREDICTION_HORIZONS_HOURS[1] - 1]),
            "target_7d": float(fc[PREDICTION_HORIZONS_HOURS[2] - 1]),
        }
        logger.info(
            "timesfm_predicted", last=round(last_price, 6), t24=round(result["target_24h"], 6)
        )
        return result
    except Exception as exc:
        logger.warning("timesfm_inference_failed", error=str(exc))
        return {"target_24h": last_price, "target_72h": last_price, "target_7d": last_price}
