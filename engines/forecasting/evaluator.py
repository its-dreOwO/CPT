"""
engines/forecasting/evaluator.py

Evaluation metrics for multi-horizon price forecasts.

Computes per-horizon metrics:
  - MAE  (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - Directional Accuracy  (fraction of correct up/down calls)
  - Sharpe Ratio          (annualised, based on predicted return)

Called by scripts/evaluate_models.py — NEVER imported by pipeline code.
"""

from __future__ import annotations

import math
import structlog
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

logger = structlog.get_logger(__name__)


@dataclass
class HorizonMetrics:
    """Evaluation results for a single prediction horizon."""

    horizon_hours: int
    mae: float
    rmse: float
    directional_accuracy: float  # [0.0, 1.0]
    sharpe: float  # annualised Sharpe on predicted returns
    n_samples: int


@dataclass
class EvaluationReport:
    """Full evaluation report across all horizons for one coin/model pair."""

    coin: str
    model: str
    horizons: dict[int, HorizonMetrics] = field(default_factory=dict)

    @property
    def mean_directional_accuracy(self) -> float:
        vals = [h.directional_accuracy for h in self.horizons.values()]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_sharpe(self) -> float:
        vals = [h.sharpe for h in self.horizons.values()]
        return float(np.mean(vals)) if vals else 0.0

    def passes_threshold(
        self,
        min_directional_accuracy: float = 0.55,
        min_sharpe: float = 1.0,
    ) -> bool:
        """Return True if model meets minimum quality bar."""
        return (
            self.mean_directional_accuracy >= min_directional_accuracy
            and self.mean_sharpe >= min_sharpe
        )

    def log_summary(self) -> None:
        logger.info(
            "eval_report",
            coin=self.coin,
            model=self.model,
            mean_dir_acc=round(self.mean_directional_accuracy, 4),
            mean_sharpe=round(self.mean_sharpe, 4),
            passes=self.passes_threshold(),
        )
        for h, m in sorted(self.horizons.items()):
            logger.info(
                "eval_horizon",
                horizon_h=h,
                mae=round(m.mae, 6),
                rmse=round(m.rmse, 6),
                dir_acc=round(m.directional_accuracy, 4),
                sharpe=round(m.sharpe, 4),
                n=m.n_samples,
            )


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    return float(math.sqrt(np.mean((actual - predicted) ** 2)))


def directional_accuracy(
    current_prices: np.ndarray,
    actual_future: np.ndarray,
    predicted_future: np.ndarray,
) -> float:
    """Fraction of samples where the predicted direction matches actual.

    Direction = sign(future - current). Ties (zero change) are excluded.
    """
    actual_dir = np.sign(actual_future - current_prices)
    pred_dir = np.sign(predicted_future - current_prices)
    mask = actual_dir != 0  # exclude zero-change rows
    if mask.sum() == 0:
        return 0.5
    return float(np.mean(actual_dir[mask] == pred_dir[mask]))


def sharpe_ratio(
    current_prices: np.ndarray,
    predicted_future: np.ndarray,
    actual_future: np.ndarray,
    horizon_hours: int,
    trading_hours_per_year: int = 8760,
) -> float:
    """Annualised Sharpe ratio of the predicted-return strategy.

    Strategy: go long if predicted return > 0, go short otherwise.
    The realised return is the actual return.

    Args:
        current_prices:    Reference prices (at prediction time).
        predicted_future:  Predicted prices at the horizon.
        actual_future:     Actual prices at the horizon.
        horizon_hours:     Horizon in hours (24, 72, 168).
        trading_hours_per_year: Used for annualisation (default: 24*365).

    Returns:
        Annualised Sharpe ratio. Returns 0.0 if std is zero.
    """
    pred_return = (predicted_future - current_prices) / (current_prices + 1e-10)
    actual_return = (actual_future - current_prices) / (current_prices + 1e-10)
    # strategy return: long if predicted up, short if predicted down
    strategy_return = np.sign(pred_return) * actual_return
    if strategy_return.std() < 1e-10:
        return 0.0
    periods_per_year = trading_hours_per_year / max(horizon_hours, 1)
    sr = (strategy_return.mean() / strategy_return.std()) * math.sqrt(periods_per_year)
    return float(sr)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def evaluate_horizon(
    coin: str,
    model: str,
    horizon_hours: int,
    current_prices: np.ndarray,
    actual_future: np.ndarray,
    predicted_future: np.ndarray,
) -> HorizonMetrics:
    """Compute all metrics for one (model, horizon) pair.

    Args:
        coin:              e.g. 'SOL'
        model:             e.g. 'xgboost'
        horizon_hours:     24, 72, or 168
        current_prices:    (N,) prices at prediction time
        actual_future:     (N,) actual prices at horizon
        predicted_future:  (N,) model-predicted prices at horizon

    Returns:
        HorizonMetrics dataclass.
    """
    assert (
        len(current_prices) == len(actual_future) == len(predicted_future)
    ), "All arrays must have the same length"
    n = len(actual_future)
    m = HorizonMetrics(
        horizon_hours=horizon_hours,
        mae=mae(actual_future, predicted_future),
        rmse=rmse(actual_future, predicted_future),
        directional_accuracy=directional_accuracy(current_prices, actual_future, predicted_future),
        sharpe=sharpe_ratio(current_prices, predicted_future, actual_future, horizon_hours),
        n_samples=n,
    )
    logger.info(
        "horizon_evaluated",
        coin=coin,
        model=model,
        horizon_h=horizon_hours,
        mae=round(m.mae, 6),
        dir_acc=round(m.directional_accuracy, 4),
        sharpe=round(m.sharpe, 4),
        n=n,
    )
    return m


def evaluate(
    coin: str,
    model: str,
    horizon_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> EvaluationReport:
    """Evaluate a model across multiple horizons.

    Args:
        coin:         e.g. 'SOL'
        model:        e.g. 'lstm'
        horizon_data: {horizon_hours: (current_prices, actual_future, predicted_future)}

    Returns:
        EvaluationReport with per-horizon metrics.
    """
    report = EvaluationReport(coin=coin, model=model)
    for h, (cur, act, pred) in sorted(horizon_data.items()):
        report.horizons[h] = evaluate_horizon(coin, model, h, cur, act, pred)
    report.log_summary()
    return report


def compare_reports(
    reports: list[EvaluationReport],
) -> Optional[EvaluationReport]:
    """Return the report with the best mean directional accuracy."""
    if not reports:
        return None
    return max(reports, key=lambda r: r.mean_directional_accuracy)
