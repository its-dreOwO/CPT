"""
engines/forecasting/ensemble.py

Weighted ensemble combiner for multi-horizon price forecasting.

Combines 5 model outputs using weights from config/constants.py:
  TimesFM=30%, TFT=25%, LSTM=20%, XGBoost=15%, LightGBM=10%

Returns a PredictionResult dataclass consumed by predictor.py and
pipeline/alert_pipeline.py.

Rules:
  - NEVER import trainer.py here. Inference only.
  - If a model is unavailable, its weight is redistributed to others.
  - Confidence is derived from model agreement (lower spread = higher confidence).
"""

from __future__ import annotations

import structlog
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from config.constants import DEFAULT_ENSEMBLE_WEIGHTS, SOL_ENSEMBLE_WEIGHTS

logger = structlog.get_logger(__name__)

# Per-coin weight tables. Coins not listed here use DEFAULT_ENSEMBLE_WEIGHTS.
_WEIGHTS_BY_COIN: dict[str, dict[str, float]] = {
    "SOL": SOL_ENSEMBLE_WEIGHTS,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Output of a single ensemble prediction run.

    Fields
    ------
    coin           : 'SOL' or 'DOGE'
    current_price  : latest known price at prediction time (USD)
    target_24h     : predicted price at +24 hours
    target_72h     : predicted price at +72 hours
    target_7d      : predicted price at +168 hours
    pct_change_24h : predicted % change for 24h horizon
    pct_change_72h : predicted % change for 72h horizon
    pct_change_7d  : predicted % change for 7d horizon
    direction_24h  : 'UP' / 'DOWN' / 'FLAT'
    confidence     : [0.0, 1.0] — based on inter-model agreement
    model_outputs  : raw per-model predictions (for logging / debugging)
    models_used    : list of model keys that contributed
    predicted_at   : UTC timestamp of prediction
    """

    coin: str
    current_price: float
    target_24h: float
    target_72h: float
    target_7d: float
    pct_change_24h: float
    pct_change_72h: float
    pct_change_7d: float
    direction_24h: str  # 'UP' / 'DOWN' / 'FLAT'
    confidence: float  # [0.0, 1.0]
    model_outputs: dict[str, dict[str, float]] = field(default_factory=dict)
    models_used: list[str] = field(default_factory=list)
    predicted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def should_alert(
        self,
        min_confidence: float = 0.70,
        min_move_pct: float = 3.0,
    ) -> bool:
        """Return True if this prediction meets the alert threshold."""
        return self.confidence >= min_confidence and abs(self.pct_change_24h) >= min_move_pct

    def to_dict(self) -> dict:
        return {
            "coin": self.coin,
            "current_price": self.current_price,
            "target_24h": round(self.target_24h, 8),
            "target_72h": round(self.target_72h, 8),
            "target_7d": round(self.target_7d, 8),
            "pct_change_24h": round(self.pct_change_24h, 4),
            "pct_change_72h": round(self.pct_change_72h, 4),
            "pct_change_7d": round(self.pct_change_7d, 4),
            "direction_24h": self.direction_24h,
            "confidence": round(self.confidence, 4),
            "models_used": self.models_used,
            "predicted_at": self.predicted_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _redistribute_weights(available: list[str], coin: str) -> dict[str, float]:
    """Return normalised weights for only the available models."""
    if not available:
        return {}
    base = _WEIGHTS_BY_COIN.get(coin, DEFAULT_ENSEMBLE_WEIGHTS)
    raw = {k: base[k] for k in available if k in base}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def _weighted_mean(values: dict[str, float], weights: dict[str, float]) -> float:
    """Compute weighted mean over model predictions."""
    total = sum(weights[k] * v for k, v in values.items() if k in weights)
    return float(total)


def _confidence_from_spread(
    predictions: dict[str, float],
    current_price: float,
) -> float:
    """Estimate confidence from inter-model spread.

    Logic:
      - Compute coefficient of variation (std / mean) across model predictions.
      - Low CV  → models agree → higher confidence.
      - High CV → models disagree → lower confidence.
      - Also apply a direction-agreement bonus: if all models agree on direction,
        add up to +0.10 to confidence.
    """
    vals = np.array(list(predictions.values()), dtype=float)
    if len(vals) == 0:
        return 0.0

    mean_pred = float(np.mean(vals))
    if abs(mean_pred) < 1e-10:
        return 0.0

    cv = float(np.std(vals) / abs(mean_pred))  # coefficient of variation
    # Map CV to [0, 1]: CV=0 → conf=1.0, CV≥0.05 → conf decays to ~0.0
    base_conf = max(0.0, min(1.0, 1.0 - cv / 0.05))

    # Direction agreement bonus
    directions = np.sign(vals - current_price)
    if len(directions) > 0:
        agreement = abs(float(np.mean(directions)))  # 1.0 = all agree
        base_conf = min(1.0, base_conf + 0.10 * agreement)

    return round(base_conf, 4)


def _direction_label(current: float, predicted: float, threshold_pct: float = 0.5) -> str:
    if current <= 0:
        return "FLAT"
    pct = (predicted - current) / current * 100
    if pct >= threshold_pct:
        return "UP"
    if pct <= -threshold_pct:
        return "DOWN"
    return "FLAT"


def _pct_change(current: float, predicted: float) -> float:
    if current <= 0:
        return 0.0
    return (predicted - current) / current * 100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def combine(
    coin: str,
    current_price: float,
    model_outputs: dict[str, dict[str, float]],
) -> PredictionResult:
    """Combine per-model predictions into a single PredictionResult.

    Args:
        coin:          'SOL' or 'DOGE'
        current_price: latest known price (USD)
        model_outputs: {model_key: {"target_24h": float, "target_72h": float, "target_7d": float}}
                       Model keys: 'timesfm', 'tft', 'lstm', 'xgboost', 'lightgbm'
                       Missing models are skipped gracefully.

    Returns:
        PredictionResult with ensemble forecasts and confidence score.
    """
    available = [k for k in model_outputs if k in _WEIGHTS]
    if not available:
        logger.error("ensemble_no_models", coin=coin)
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
            model_outputs=model_outputs,
            models_used=[],
        )

    weights = _redistribute_weights(available, coin)

    t24 = _weighted_mean({k: model_outputs[k]["target_24h"] for k in available}, weights)
    t72 = _weighted_mean({k: model_outputs[k]["target_72h"] for k in available}, weights)
    t7d = _weighted_mean({k: model_outputs[k]["target_7d"] for k in available}, weights)

    # Confidence based on 24h predictions (primary signal)
    preds_24h = {k: model_outputs[k]["target_24h"] for k in available}
    confidence = _confidence_from_spread(preds_24h, current_price)

    result = PredictionResult(
        coin=coin,
        current_price=current_price,
        target_24h=t24,
        target_72h=t72,
        target_7d=t7d,
        pct_change_24h=_pct_change(current_price, t24),
        pct_change_72h=_pct_change(current_price, t72),
        pct_change_7d=_pct_change(current_price, t7d),
        direction_24h=_direction_label(current_price, t24),
        confidence=confidence,
        model_outputs=model_outputs,
        models_used=available,
    )

    logger.info(
        "ensemble_combined",
        coin=coin,
        models=available,
        weights={k: round(v, 3) for k, v in weights.items()},
        t24=round(t24, 6),
        pct_24h=round(result.pct_change_24h, 3),
        direction=result.direction_24h,
        confidence=confidence,
    )
    return result
